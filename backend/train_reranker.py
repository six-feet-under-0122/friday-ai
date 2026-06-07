# train_reranker.py
import os
import json
import pickle
from pathlib import Path
from collections import Counter

import numpy as np
from langchain_core.documents import Document

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from sklearn.metrics import classification_report, accuracy_score

import svm_reranker
import rag_core


DATA_PATH = "data/labeled.jsonl"
MODEL_DIR = "models"
MODEL_PATH = "models/reranker_pipeline.pkl"


def load_labeled_data(path: str):
    """
    读取人工标注后的 labeled.jsonl。
    只保留 label 为 0/1 的样本。
    """
    rows = []

    if not os.path.exists(path):
        raise FileNotFoundError(f"找不到标注数据文件: {path}")

    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue

            obj = json.loads(line)
            label = obj.get("label", None)

            # 有些 JSON 里可能是 None，有些可能是字符串
            if label is None or label == "":
                continue

            label = int(label)
            if label not in (0, 1):
                continue

            rows.append(obj)

    return rows


def build_features(rows):
    """
    把每一条 query + chunk 样本转换成 SVM 可以吃的特征向量 X。
    标签 y 就是你手工标的 0/1。
    """
    embeddings = rag_core._get_embeddings()

    X = []
    y = []

    # 避免同一个 query 重复算 embedding
    query_emb_cache = {}

    # 避免同一个 chunk 重复算 embedding
    doc_emb_cache = {}

    for idx, r in enumerate(rows, start=1):
        query = r.get("query", "")
        chunk_text = r.get("chunk_text", "")
        label = int(r.get("label"))

        bm25_rank = int(r.get("bm25_rank", 10000))
        vec_rank = int(r.get("vec_rank", 10000))

        if not query or not chunk_text:
            continue

        # query embedding
        if query not in query_emb_cache:
            query_emb_cache[query] = embeddings.embed_query(query)

        q_emb = np.array(query_emb_cache[query], dtype=np.float32)

        # document embedding
        # 注意：训练时 labeled.jsonl 里未必保存了 emb，所以这里重新算一遍
        chunk_id = str(r.get("chunk_id", ""))
        doc_key = chunk_id + "||" + chunk_text[:80]

        if doc_key not in doc_emb_cache:
            doc_emb_cache[doc_key] = embeddings.embed_query(chunk_text)

        d_emb = doc_emb_cache[doc_key]

        doc = Document(
            page_content=chunk_text,
            metadata={
                "chunk_id": chunk_id,
                "page": r.get("page", None),
                "emb": d_emb,
            }
        )

        features = svm_reranker.extract_features(
            query=query,
            doc=doc,
            bm25_rank=bm25_rank,
            vec_rank=vec_rank,
            q_emb=q_emb,
        )

        X.append(features)
        y.append(label)

        if idx % 20 == 0:
            print(f"已处理 {idx}/{len(rows)} 条样本")

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y


def main():
    print("========== 开始训练 SVM Reranker ==========")

    rows = load_labeled_data(DATA_PATH)
    print(f"读取到有效标注样本数: {len(rows)}")

    if len(rows) == 0:
        raise RuntimeError("没有读取到任何有效样本，请检查 data/labeled.jsonl 里的 label 是否为 0/1。")

    label_counter = Counter([int(r["label"]) for r in rows if r.get("label") is not None])
    print(f"标签分布: {label_counter}")

    if len(label_counter) < 2:
        raise RuntimeError(
            "你的标注数据只有一个类别。SVM 至少需要同时有 label=0 和 label=1 的样本。"
        )

    X, y = build_features(rows)

    print("特征矩阵形状:", X.shape)
    print("标签形状:", y.shape)

    feature_names = [
        "bm25_rank_score",
        "vec_rank_score",
        "cos_sim",
        "overlap_cnt",
        "overlap_ratio",
        "jaccard",
        "chunk_len",
    ]

    print("\n特征名称:")
    for i, name in enumerate(feature_names):
        print(f"{i}: {name}")

    # 如果样本太少，test_size 可以调小
    test_size = 0.2
    stratify = y if min(Counter(y).values()) >= 2 else None

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=42,
        stratify=stratify
    )

    print("\n训练集大小:", X_train.shape)
    print("测试集大小:", X_test.shape)

    # SVM Pipeline
    # StandardScaler: 标准化特征
    # LinearSVC: 线性支持向量机
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        ("svm", LinearSVC(
            class_weight="balanced",
            random_state=42,
            max_iter=10000
        ))
    ])

    pipe.fit(X_train, y_train)

    y_pred = pipe.predict(X_test)

    print("\n========== 测试集评估结果 ==========")
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print(classification_report(y_test, y_pred, digits=4))

    # 查看线性 SVM 的特征权重
    svm = pipe.named_steps["svm"]
    weights = svm.coef_[0]

    print("\n========== 特征权重 ==========")
    for name, w in sorted(zip(feature_names, weights), key=lambda x: abs(x[1]), reverse=True):
        print(f"{name}: {w:.4f}")

    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as f:
        pickle.dump(pipe, f)

    print("\n✅ 模型训练完成！")
    print(f"✅ 已保存到: {MODEL_PATH}")
    print("之后 app.py 调用 /chat 时，会自动加载这个 reranker 模型。")


if __name__ == "__main__":
    main()
