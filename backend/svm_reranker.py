# svm_reranker.py
import os
import re
import pickle
import numpy as np
import jieba
from typing import List, Dict
from langchain_core.documents import Document

MODEL_PATH = os.getenv("RERANKER_MODEL_PATH", "models/reranker_pipeline.pkl")

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

def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = (np.linalg.norm(a) * np.linalg.norm(b))
    if denom == 0:
        return 0.0
    return float(np.dot(a, b) / denom)

def load_pipeline():
    if not os.path.exists(MODEL_PATH):
        return None
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)

def extract_features(
    query: str,
    doc: Document,
    *,
    bm25_rank: int,
    vec_rank: int,
    q_emb: np.ndarray,
) -> List[float]:
    md = doc.metadata or {}
    d_emb_list = md.get("emb")  # 来自 chunks.pkl 的持久化向量
    if d_emb_list is None:
        # 如果历史 chunks 没 emb（老数据），给 0（或者你也可以直接跳过 rerank 回退）
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
    scores = pipe.decision_function(X)

    ranked = sorted(zip(scores, candidates), key=lambda x: x[0], reverse=True)
    return [d for _, d in ranked[:top_k]]