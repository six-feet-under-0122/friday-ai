import csv, json, os, re
from pathlib import Path

IN_PATH = "data/to_label.jsonl"
OUT_PATH = "data/to_label.csv"

def preview(text: str, n=120):
    text = (text or "").replace("\n", " ").replace("\r", " ")
    text = re.sub(r"\s+", " ", text).strip()
    return text[:n]

def main():
    os.makedirs("uploads/data", exist_ok=True)

    rows = []
    with open(IN_PATH, "r", encoding="utf-8") as f:
        for line in f:
            r = json.loads(line)
            rows.append({
                "query_id": r.get("query_id",""),
                "query": r.get("query",""),
                "chunk_id": r.get("chunk_id",""),
                "page": r.get("page", ""),
                "bm25_rank": r.get("bm25_rank",""),
                "vec_rank": r.get("vec_rank",""),
                "chunk_preview": preview(r.get("chunk_text","")),
                "label": r.get("label", ""),  # 空着
            })

    with open(OUT_PATH, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    print(f"✅ wrote {len(rows)} rows to {OUT_PATH}")

if __name__ == "__main__":
    main()