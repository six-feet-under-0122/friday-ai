import os
import json
from pathlib import Path

import rag_core

OUT_PATH = "../data/to_label.jsonl"
QUERY_PATH = "../data/queries.txt"

CANDIDATE_K = 20   # 每个 query 导出多少候选（你可以改 10/20/30）

def doc_key(d):
    cid = (d.metadata or {}).get("chunk_id")
    if cid:
        return str(cid)
    md = d.metadata or {}
    return f"{md.get('source', '?')}|{md.get('page', '?')}|{(d.page_content or '')[:80]}"

def main():
    os.makedirs("../data", exist_ok=True)

    bm25, vec = rag_core._get_retrievers()
    if not bm25 or not vec:
        raise RuntimeError("retrievers not ready. 请先上传/构建知识库，确保 chunks.pkl + chroma_db 存在。")

    # 确保候选池足够大（否则导出来不够）
    bm25.k = max(bm25.k, CANDIDATE_K)
    # vec 的 k 在创建 retriever 时定了；如果你要严格改，需要去 rag_core 里把 search_kwargs 改大

    queries = Path(QUERY_PATH).read_text(encoding="utf-8").splitlines()
    queries = [q.strip() for q in queries if q.strip()]
    if not queries:
        raise RuntimeError("data/queries.txt 为空")

    rows = []
    for qi, q in enumerate(queries, start=1):
        bm25_docs = bm25.invoke(q)
        vec_docs = vec.invoke(q)

        bm25_rank = {doc_key(d): r for r, d in enumerate(bm25_docs)}
        vec_rank = {doc_key(d): r for r, d in enumerate(vec_docs)}

        candidates = rag_core.weighted_hybrid_retrieve(
            q, bm25=bm25, vec=vec, k=CANDIDATE_K, w_bm25=0.5, w_vec=0.5
        )

        for d in candidates:
            k = doc_key(d)
            rows.append({
                "query_id": f"q{qi:03d}",
                "query": q,
                "chunk_id": (d.metadata or {}).get("chunk_id", ""),
                "page": (d.metadata or {}).get("page", None),
                "chunk_text": d.page_content,
                "bm25_rank": int(bm25_rank.get(k, 10_000)),
                "vec_rank": int(vec_rank.get(k, 10_000)),
                "label": None  # 你手工改成 0/1
            })

    with open(OUT_PATH, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"✅ 已导出 {len(rows)} 条待标注样本到 {OUT_PATH}")
    print("下一步：把 label 从 None 改成 0/1，另存为 data/labeled.jsonl")

if __name__ == "__main__":
    main()