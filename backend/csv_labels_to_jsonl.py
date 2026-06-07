import csv
import json
import os
from pathlib import Path

# 自动定位 backend 目录：脚本在 backend 或 backend/uploads 都能跑
BASE_DIR = Path(__file__).resolve().parent
if not (BASE_DIR / "data").exists() and (BASE_DIR.parent / "data").exists():
    BASE_DIR = BASE_DIR.parent

CSV_PATH = BASE_DIR / "data" / "to_label.csv"          # 你标注后的 CSV
JSONL_IN = BASE_DIR / "data" / "to_label.jsonl"        # 原始完整 JSONL
JSONL_OUT = BASE_DIR / "data" / "labeled.jsonl"        # 输出：完整 + label

def key(qid: str, cid: str) -> str:
    return f"{qid}||{cid}"

def main():
    if not CSV_PATH.exists():
        raise FileNotFoundError(f"CSV not found: {CSV_PATH}")
    if not JSONL_IN.exists():
        raise FileNotFoundError(f"JSONL not found: {JSONL_IN}")

    # 1) 读取 CSV 中的 label
    label_map = {}
    with open(CSV_PATH, "r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        required = {"query_id", "chunk_id", "label"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"CSV 缺少列: {missing}，当前列: {reader.fieldnames}")

        for row in reader:
            qid = (row.get("query_id") or "").strip()
            cid = (row.get("chunk_id") or "").strip()
            lab = (row.get("label") or "").strip()

            if not qid or not cid:
                continue

            # 允许空 label（表示你没标到那条）
            if lab == "":
                continue

            if lab not in ("0", "1"):
                raise ValueError(f"label 必须是 0/1（或留空），发现: {lab} (query_id={qid}, chunk_id={cid})")

            label_map[key(qid, cid)] = int(lab)

    # 2) 回写到 JSONL（保留完整 chunk_text 等字段）
    os.makedirs(BASE_DIR / "data", exist_ok=True)
    total = 0
    filled = 0
    with open(JSONL_IN, "r", encoding="utf-8") as fin, open(JSONL_OUT, "w", encoding="utf-8") as fout:
        for line in fin:
            obj = json.loads(line)
            total += 1
            qid = str(obj.get("query_id", "")).strip()
            cid = str(obj.get("chunk_id", "")).strip()
            k = key(qid, cid)

            if k in label_map:
                obj["label"] = label_map[k]
                filled += 1

            fout.write(json.dumps(obj, ensure_ascii=False) + "\n")

    print(f"✅ 输出完成: {JSONL_OUT}")
    print(f"   JSONL 总行数: {total}")
    print(f"   成功写入 label 行数: {filled}")
    print(f"   CSV 中 label 数量: {len(label_map)}")

    if filled == 0:
        print("⚠️ 没有写入任何 label：请检查 CSV 的 query_id/chunk_id 是否与 JSONL 一致。")

if __name__ == "__main__":
    main()