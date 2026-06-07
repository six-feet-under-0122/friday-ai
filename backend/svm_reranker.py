import os
import re
import pickle
import numpy as np
import jieba
from typing import List, Dict
from langchain_core.documents import Document

MODEL_PATH = "models/reranker_pipeline.pkl"

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


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """计算余弦相似度"""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def char_overlap(query: str, chunk: str) -> float:
    """字符级重叠率"""
    if not query or not chunk:
        return 0
    query_chars = set(query)
    chunk_chars = set(chunk)
    if not query_chars:
        return 0
    overlap = len(query_chars & chunk_chars)
    return overlap / len(query_chars)


def extract_features(
        query: str,
        doc: Document,
        bm25_rank: int,
        vec_rank: int,
        query_emb: np.ndarray,
        feature_cols: List[str]
) -> List[float]:
    """
    提取特征向量（必须与训练时一致）
    """
    # 获取文档embedding（从metadata中）
    md = doc.metadata or {}
    d_emb_list = md.get("emb")

    if d_emb_list is None:
        # 如果没有保存embedding，实时计算（但不推荐）
        d_emb = np.zeros_like(query_emb)
        cos_sim = 0.0
    else:
        d_emb = np.array(d_emb_list, dtype=np.float32)
        cos_sim = cosine_similarity(query_emb, d_emb)

    # BM25特征
    bm25_score = 1.0 / (bm25_rank + 1)
    bm25_rank_reciprocal = 1.0 / (bm25_rank + 10)

    # 向量特征
    vec_recalled = 1 if vec_rank < 9999 else 0
    vec_score = 1.0 / (vec_rank + 1) if vec_recalled else 0.01

    # 文本重叠
    overlap = char_overlap(query, doc.page_content)

    # 长度特征
    query_len = len(query)
    chunk_len = len(doc.page_content)
    len_ratio = chunk_len / (query_len + 1)

    # 按训练时的顺序返回特征
    feature_map = {
        'bm25_score': bm25_score,
        'bm25_rank_reciprocal': bm25_rank_reciprocal,
        'vec_recalled': float(vec_recalled),
        'vec_score': vec_score,
        'char_overlap': overlap,
        'len_ratio': len_ratio
    }

    return [feature_map[col] for col in feature_cols]


def rerank_binary(
        query: str,
        candidates: List[Document],
        bm25_docs: List[Document],
        vec_docs: List[Document],
        query_emb: List[float],
        top_k: int,
) -> List[Document]:
    """
    使用训练好的模型对候选文档重排序
    """
    pipeline = load_pipeline()

    # 如果没有模型，直接返回前top_k个
    if pipeline is None:
        return candidates[:top_k]

    model = pipeline['model']
    scaler = pipeline['scaler']
    feature_cols = pipeline['feature_cols']

    q_emb = np.array(query_emb, dtype=np.float32)

    # 构建排名映射
    bm25_rank_map = {doc_key(d): r for r, d in enumerate(bm25_docs)}
    vec_rank_map = {doc_key(d): r for r, d in enumerate(vec_docs)}
    BAD_RANK = 10000

    # 提取特征
    X = []
    for doc in candidates:
        key = doc_key(doc)
        features = extract_features(
            query, doc,
            bm25_rank=bm25_rank_map.get(key, BAD_RANK),
            vec_rank=vec_rank_map.get(key, BAD_RANK),
            query_emb=q_emb,
            feature_cols=feature_cols
        )
        X.append(features)

    # 标准化和预测
    X_scaled = scaler.transform(np.array(X, dtype=np.float32))
    scores = model.predict_proba(X_scaled)[:, 1]  # 获取"相关"的概率

    # 按分数排序
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in ranked[:top_k]]