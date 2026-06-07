# train_reranker.py (重写这个文件)
import pandas as pd
import numpy as np
import json
import pickle
import os
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, roc_auc_score, f1_score
from sklearn.model_selection import train_test_split
import warnings

warnings.filterwarnings('ignore')


def train_reranker():
    """训练重排模型并保存到 models/ 目录"""

    # 1. 加载数据
    print("=" * 60)
    print("加载标注数据...")

    data = []
    with open('data/labeled.jsonl', 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                data.append(json.loads(line))

    df = pd.DataFrame(data)
    print(f"加载完成: {len(df)} 条, 正例: {(df['label'] == 1).sum()}")

    # 2. 特征提取（使用与推理时相同的特征）
    print("\n提取特征...")

    # BM25特征
    df['bm25_score'] = 1.0 / (df['bm25_rank'] + 1)
    df['bm25_rank_reciprocal'] = 1.0 / (df['bm25_rank'] + 10)

    # 向量特征（处理未召回）
    df['vec_recalled'] = (df['vec_rank'] < 9999).astype(int)
    df['vec_score'] = np.where(df['vec_recalled'] == 1, 1.0 / (df['vec_rank'] + 1), 0.01)

    # 文本重叠特征
    def char_overlap(query, chunk):
        if not query or not chunk:
            return 0
        query_chars = set(query)
        chunk_chars = set(chunk)
        if not query_chars:
            return 0
        overlap = len(query_chars & chunk_chars)
        return overlap / len(query_chars)

    df['char_overlap'] = df.apply(
        lambda row: char_overlap(row['query'], row['chunk_text']),
        axis=1
    )

    # 长度特征
    df['query_len'] = df['query'].apply(len)
    df['chunk_len'] = df['chunk_text'].apply(len)
    df['len_ratio'] = df['chunk_len'] / (df['query_len'] + 1)

    # 3. 准备训练数据
    feature_cols = [
        'bm25_score', 'bm25_rank_reciprocal',
        'vec_recalled', 'vec_score',
        'char_overlap',
        'len_ratio'
    ]

    X = df[feature_cols].values
    y = df['label'].values

    # 按query划分
    unique_queries = df['query_id'].unique()
    train_queries, test_queries = train_test_split(unique_queries, test_size=0.25, random_state=42)

    train_mask = df['query_id'].isin(train_queries)
    test_mask = df['query_id'].isin(test_queries)

    X_train, X_test = X[train_mask], X[test_mask]
    y_train, y_test = y[train_mask], y[test_mask]

    print(f"训练集: {X_train.shape[0]} 条, 测试集: {X_test.shape[0]} 条")

    # 4. 标准化
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5. 训练逻辑回归（简单有效）
    model = LogisticRegression(
        class_weight='balanced',
        C=1.0,
        max_iter=1000,
        random_state=42
    )

    model.fit(X_train_scaled, y_train)

    # 6. 评估
    y_pred = model.predict(X_test_scaled)
    y_prob = model.predict_proba(X_test_scaled)[:, 1]

    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)

    print(f"\n模型评估:")
    print(f"AUC-ROC: {auc:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=['不相关', '相关']))

    # 7. 保存模型
    os.makedirs('models', exist_ok=True)

    pipeline = {
        'model': model,
        'scaler': scaler,
        'feature_cols': feature_cols,
        'metrics': {'auc': auc, 'f1': f1}
    }

    with open('models/reranker_pipeline.pkl', 'wb') as f:
        pickle.dump(pipeline, f)

    print("\n✅ 模型已保存到 models/reranker_pipeline.pkl")

    # 8. 特征重要性
    feature_importance = np.abs(model.coef_[0])
    importance_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': feature_importance
    }).sort_values('importance', ascending=False)

    print("\n特征重要性:")
    print(importance_df)

    return model, scaler, feature_cols


if __name__ == "__main__":
    train_reranker()