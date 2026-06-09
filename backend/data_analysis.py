import pickle
import os
import jieba
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# 对应你 rag_core.py 里保存的切块文件
CHUNKS_FILE = "chunks.pkl"


def generate_analysis_report():
    print("开始生成数据集分析报告...")

    if not os.path.exists(CHUNKS_FILE):
        return {"keywords": [], "clusters": []}

    with open(CHUNKS_FILE, "rb") as f:
        chunks = pickle.load(f)

    if not chunks:
        return {"keywords": [], "clusters": []}

    # 1. 提取所有文本块
    texts = [doc.page_content for doc in chunks if doc.page_content]

    # 2. 数据预处理：中文分词与停用词过滤
    def tokenize(text):
        words = jieba.lcut(text)
        # 过滤掉单字和无意义的符号，只保留长度 >= 2 的词
        return " ".join([w for w in words if len(w.strip()) >= 2])

    corpus = [tokenize(t) for t in texts]

    if not corpus:
        return {"keywords": [], "clusters": []}

    # 3. 统计分析：计算 TF-IDF 并提取 Top 15 关键词
    vectorizer = TfidfVectorizer(max_features=500)
    tfidf_matrix = vectorizer.fit_transform(corpus)

    feature_names = vectorizer.get_feature_names_out()
    # 计算每个词的平均 TF-IDF 权重
    avg_tfidf = np.asarray(tfidf_matrix.mean(axis=0)).ravel()
    top_indices = avg_tfidf.argsort()[::-1][:15]  # 取前 15 个最大的

    keywords = [
        {"name": feature_names[i], "value": round(float(avg_tfidf[i]), 4)}
        for i in top_indices
    ]

    # 4. 统计分析：K-Means 文本聚类
    # 假设我们将文本块分成 4 个主题（如果切块少于4个，就按实际数量聚类）
    n_clusters = min(4, len(texts))
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    kmeans.fit(tfidf_matrix)

    labels = kmeans.labels_
    cluster_counts = {}
    for label in labels:
        cluster_counts[label] = cluster_counts.get(label, 0) + 1

    # 生成前端需要的饼图数据格式
    clusters = []
    for i in range(n_clusters):
        clusters.append({
            "name": f"内容簇 {i + 1}",
            "value": cluster_counts.get(i, 0)
        })

    print("分析完成！")
    return {
        "keywords": keywords,
        "clusters": clusters
    }