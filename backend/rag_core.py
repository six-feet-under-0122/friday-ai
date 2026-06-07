from dotenv import load_dotenv
import os

load_dotenv()
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from typing import List, Dict
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
# 下面的import 可能因版本不同需调整
from langchain_community.embeddings import ZhipuAIEmbeddings

import pickle
from functools import lru_cache
from langchain_community.retrievers import BM25Retriever

import re
import jieba


import svm_reranker
PERSIST_DIR = "chroma_db"
CHUNKS_FILE = "chunks.pkl"



def load_pdf_pages(pdf_path: str):
    """加载 PDF 文件并进行严格的路径与内容校验"""
    pdf_path_obj = Path(pdf_path).expanduser().resolve()

    if not pdf_path_obj.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path_obj}")

    loader = PyMuPDFLoader(str(pdf_path_obj))
    docs = loader.load()

    if not docs:
        raise ValueError(f"Loaded 0 pages from PDF: {pdf_path_obj}")

    return docs


def chunk_documents(page_docs, chunk_size: int = 800, chunk_overlap: int = 120):
    """将文档按页切块，并注入增强的元数据(metadata)"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )

    chunks = []
    for d in page_docs:

        page_text = (d.page_content or "").strip()
        if not page_text:
            continue

        pieces = splitter.split_text(page_text)

        for j, piece in enumerate(pieces):
            md = dict(d.metadata or {})
            md["chunk_id"] = f"p{md.get('page', '?')}_c{j}"
            md["chunk_size"] = len(piece)
            chunks.append(Document(page_content=piece, metadata=md))

    return chunks


# ---------- 3. 核心主流程 (供 Flask 调用) ----------
def process_pdf(file_path):
    print(f"开始处理文件: {file_path}")

    try:
        documents = load_pdf_pages(file_path)
        print(f"✅ PDF 加载完成，共 {len(documents)} 页。")

        chunked_docs = chunk_documents(documents, chunk_size=800, chunk_overlap=120)
        print(f"✅ 文本切块完成，一共切出了 {len(chunked_docs)} 个文本块！")

        embeddings = _get_embeddings()

        # ===== 新增：给每个 chunk 计算 embedding 并存入 metadata =====
        texts = [d.page_content for d in chunked_docs]
        vecs = embeddings.embed_documents(texts)  # list[list[float]]

        for d, v in zip(chunked_docs, vecs):
            md = d.metadata or {}
            md["emb"] = v               # 关键：持久化向量
            d.metadata = md

        # ===== 保存 chunks =====
        all_chunks = _load_chunks()
        all_chunks.extend(chunked_docs)
        _save_chunks(all_chunks)
        _get_retrievers.cache_clear()

        # ===== 向量库仍然照旧建立（用于 vec 召回）=====
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory="chroma_db"
        )

        print("✅ 存入数据库成功！MVP 知识库构建完毕！")
        return True

    except Exception as e:
        print(f"❌ 处理 PDF 失败: {e}")
        raise e

# 新加的代码----------------------------------------------------------------------------------
# ----------------------------新加的代码----------------------------------------

def _doc_key(d: Document) -> str:
    # 优先用你在切分时写入的 chunk_id
    cid = (d.metadata or {}).get("chunk_id")
    if cid:
        return str(cid)

        # 兜底：source + page（再兜底取文本前缀）
    md = d.metadata or {}
    return f"{md.get('source', '?')}|{md.get('page', '?')}|{(d.page_content or '')[:80]}"


def weighted_hybrid_retrieve(
        query: str,
        *,
        bm25,
        vec,
        k: int = 8,
        w_bm25: float = 0.5,
        w_vec: float = 0.5,
) -> List[Document]:
    bm25_docs = bm25.invoke(query)

    vec_docs = vec.invoke(query)

    scores: Dict[str, float] = {}
    picked: Dict[str, Document] = {}

    def add(docs: List[Document], weight: float):
        for rank, d in enumerate(docs):
            key = _doc_key(d)
            picked[key] = d
            scores[key] = scores.get(key, 0.0) + weight * (1.0 / (rank + 1))

    add(bm25_docs, w_bm25)
    add(vec_docs, w_vec)

    ranked_keys = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [picked[kk] for kk in ranked_keys[:k]]

#新增持久化 + hybrid 检索入口
def _load_chunks():
    if os.path.exists(CHUNKS_FILE):
        with open(CHUNKS_FILE, "rb") as f:
            return pickle.load(f)
    return []

def _save_chunks(chunks):
    with open(CHUNKS_FILE, "wb") as f:
        pickle.dump(chunks, f)

def _get_embeddings():
    return ZhipuAIEmbeddings(
        api_key=os.getenv("ZHIPUAI_API_KEY"),
        model="embedding-3"
    )

def _get_vectorstore():
    embeddings = _get_embeddings()
    if Path(PERSIST_DIR).exists():
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            embedding_function=embeddings
        )
        return vs
    else:
        return None

@lru_cache(maxsize=1)
def _get_retrievers():
    # load chunks
    chunks = _load_chunks()
    if not chunks:
        return None, None

    # bm25
    bm25 = BM25Retriever.from_documents(chunks, preprocess_func=zh_tokenize)
    bm25.k = 30



    # vector
    vs = _get_vectorstore()
    if vs is None:
        return bm25, None
    vec = vs.as_retriever(search_kwargs={"k": 30})

    return bm25, vec

def retrieve_context(query: str, k: int = 5) -> str:
    bm25, vec = _get_retrievers()
    if not bm25 or not vec:
        return ""

    hits = weighted_hybrid_retrieve(query, bm25=bm25, vec=vec, k=k)
    if not hits:
        return ""

    # 拼成一段 context
    context = "\n\n".join(
        [f"[{i+1}] {d.page_content}" for i, d in enumerate(hits)]
    )
    return context



def retrieve_with_sources(query: str, k: int = 5):
    bm25, vec = _get_retrievers()
    if not bm25 or not vec:
        return "", []

    candidate_k = 30

    # 为了 rank 特征，拿到各自的排序列表
    bm25_docs = bm25.invoke(query)
    vec_docs = vec.invoke(query)

    # 先 hybrid 融合出候选池
    candidates = weighted_hybrid_retrieve(query, bm25=bm25, vec=vec, k=candidate_k)
    if not candidates:
        return "", []

    # query embedding（只算一次）
    embeddings = _get_embeddings()
    q_emb = embeddings.embed_query(query)

    # SVM 重排
    hits = svm_reranker.rerank_binary(
        query=query,
        candidates=candidates,
        bm25_docs=bm25_docs,
        vec_docs=vec_docs,
        query_emb=q_emb,
        top_k=k,
    )

    context = "\n\n".join([f"[{i+1}] {d.page_content}" for i, d in enumerate(hits)])
    sources = [
        {"filename": d.metadata.get("source_filename", ""),
         "page": d.metadata.get("page", None),
         "chunk_id": d.metadata.get("chunk_id", "")}
        for d in hits
    ]
    return context, sources

def zh_tokenize(text: str):
    text = (text or "").strip().lower()
    text = re.sub(r"\s+", " ", text)
    tokens = list(jieba.cut(text))   # 兼容性最好
    print(tokens)
    return [t for t in tokens if t.strip()]