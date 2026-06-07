# svm_reranker.py
import os
import re
import pickle
import numpy as np
import jieba
from typing import List
from langchain_core.documents import Document

MODEL_PATH = "models/reranker_pipeline.pkl"

# 全局缓存
_pipeline_cache = None


def load_pipeline():
    global _pipeline_cache
    if _pipeline_cache is not None:
        return _pipeline_cache

    if not os.path.exists(MODEL_PATH):
        print(f"⚠️ 模型文件不存在: {MODEL_PATH}")
        return None

    with open(MODEL_PATH, "rb") as f:
        _pipeline_cache = pickle.load(f)

    print("✅ 重排模型加载成功")
    return _pipeline_cache


def zh_tokenize_simple(text: str):
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    return [t for t in jieba.cut(text) if t.strip()]


def doc_key(d: Document) -> str:
    cid = (d.metadata or {}).get("chunk_id")
    if cid:
        return str(cid)
    md = d.metadata or {}
    return f"{md.get('source', '?')}|{md.get('page', '?')}|{(d.page_content or '')[:80]}"


def char_overlap(query: str, chunk: str) -> float:
    """字符级重叠率（必须与训练时完全一致）"""
    if not query or not chunk:
        return 0
    query_chars = set(query)
    chunk_chars = set(chunk)
    if not query_chars:
        return 0
    overlap = len(query_chars & chunk_chars)
    return overlap / len(query_chars)


def extract_features(query: str, doc: Document, bm25_rank: int, vec_rank: int) -> List[float]:
    """
    提取特征向量
    ⚠️ 特征顺序必须与训练时完全一致
    """
    # BM25特征
    bm25_score = 1.0 / (bm25_rank + 1)
    bm25_rank_reciprocal = 1.0 / (bm25_rank + 10)

    # 向量特征
    vec_recalled = 1.0 if vec_rank < 9999 else 0.0
    vec_score = 1.0 / (vec_rank + 1) if vec_recalled else 0.01

    # 文本重叠
    overlap = char_overlap(query, doc.page_content)

    # 长度特征
    query_len = len(query)
    chunk_len = len(doc.page_content)
    len_ratio = chunk_len / (query_len + 1) if query_len > 0 else 1.0

    # ⚠️ 顺序必须与训练时的 feature_cols 一致
    # ['bm25_score', 'bm25_rank_reciprocal', 'vec_recalled', 'vec_score', 'char_overlap', 'len_ratio']
    features = [
        bm25_score,
        bm25_rank_reciprocal,
        vec_recalled,
        vec_score,
        overlap,
        len_ratio
    ]

    return features


def rerank_binary(
        query: str,
        candidates: List[Document],
        bm25_docs: List[Document],
        vec_docs: List[Document],
        query_emb: List[float],  # 保留参数但当前版本不使用
        top_k: int,
) -> List[Document]:
    """
    使用训练好的模型对候选文档重排序
    """
    pipeline = load_pipeline()

    if pipeline is None:
        return candidates[:top_k]

    model = pipeline['model']
    scaler = pipeline['scaler']

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
            vec_rank=vec_rank_map.get(key, BAD_RANK)
        )
        X.append(features)

    # 标准化和预测
    X_scaled = scaler.transform(np.array(X, dtype=np.float32))
    scores = model.predict_proba(X_scaled)[:, 1]

    # 按分数排序
    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)

    return [doc for _, doc in ranked[:top_k]]