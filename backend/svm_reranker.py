# svm_reranker.py
import os
import re
import pickle
import numpy as np
import jieba
from typing import List, Dict
from langchain_core.documents import Document

MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "models/reranker_pipeline.pkl")

# 全局缓存模型
_pipeline_cache = None


def load_pipeline():
    """加载训练好的重排模型"""
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ 模型文件不存在: {MODEL_PATH}，将使用原始排序")
        return None

    with open(MODEL_PATH, "rb") as f:
        _pipeline_cache = pickle.load(f)

    print("✅ 重排模型加载成功")
    return _pipeline_cache


def zh_tokenize_simple(text: str):
    """简单分词"""
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return [t for t in jieba.cut(text) if t.strip()]


def doc_key(d: Document) -> str:
    """生成文档唯一标识"""
    cid = (d.metadata or {}).get("chunk_id")
    if cid:
        return str(cid)
    md = d.metadata or {}
    return f"{md.get('source', '?')}|{md.get('page', '?')}|{(d.page_content or '')[:80]}"


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)


def extract_features(
        query: str,
        doc: Document,
        *,
        bm25_rank: int,
        vec_rank: int,
        q_emb: np.ndarray,
) -> List[float]:
    """
    提取 7 个特征（必须与 train_reranker.py 严格一致）
    """
    md = doc.metadata or {}
    d_emb_list = md.get("emb")

    if d_emb_list is None:
        d_emb = np.zeros_like(q_emb)
    else:
        d_emb = np.array(d_emb_list, dtype=np.float32)

    cos_sim = cosine(q_emb, d_emb)

    q_tokens = set(zh_tokenize_simple(query))
    d_tokens = set(zh_tokenize_simple(doc.page_content))

    inter = q_tokens & d_tokens
    union = q_tokens | d_tokens

    overlap_cnt = len(inter)
    overlap_ratio = overlap_cnt / max(1, len(q_tokens))
    jaccard = len(inter) / max(1, len(union))

    chunk_len = len(doc.page_content or "")

    bm25_rank_score = 1.0 / (bm25_rank + 1)
    vec_rank_score = 1.0 / (vec_rank + 1)

    # 这里的顺序必须和训练时完全一致！
    return [
        bm25_rank_score,
        vec_rank_score,
        cos_sim,
        overlap_cnt,
        overlap_ratio,
        jaccard,
        chunk_len,
    ]


def rerank_binary(
        query: str,
        candidates: List[Document],
        *,
        bm25_docs: List[Document],
        vec_docs: List[Document],
        query_emb: List[float],
        top_k: int,
) -> List[Document]:
    """
    使用训练好的模型对候选文档重排序
    """
    pipe = load_pipeline()

    if pipe is None:
        return candidates[:top_k]

    q_emb = np.array(query_emb, dtype=np.float32)

    bm25_rank_map = {doc_key(d): r for r, d in enumerate(bm25_docs)}
    vec_rank_map = {doc_key(d): r for r, d in enumerate(vec_docs)}
    bad_rank = 10_000

    X = []
    for d in candidates:
        k = doc_key(d)
        X.append(
            extract_features(
                query, d,
                bm25_rank=bm25_rank_map.get(k, bad_rank),
                vec_rank=vec_rank_map.get(k, bad_rank),
                q_emb=q_emb,
            )
        )

    X = np.array(X, dtype=np.float32)

    # 【重点】：我们保存的 pipe 是一个完整的 Pipeline，它内部自动包含 scaler(归一化) 和 svm(打分)
    # 所以直接调用 pipe.decision_function(X) 即可
    scores = pipe.decision_function(X)

    # 按得分从高到低排序
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [d for _, d in ranked[:top_k]]