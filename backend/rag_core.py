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
            md["chunk_id"] = f"p{md.get('page','?')}_c{j}"
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
        if chunked_docs:
            print(f"   [预览] chunk[0] metadata: {chunked_docs[0].metadata}")
            print(f"   [预览] chunk[0] content: {chunked_docs[0].page_content[:100]}...\n")


        embeddings = ZhipuAIEmbeddings(
            api_key=os.getenv("ZHIPUAI_API_KEY"),
            model="embedding-3"  
        )
        
        vectorstore = Chroma.from_documents(
            documents=chunked_docs,
            embedding=embeddings,
            persist_directory="chroma_db"
        )
        
        print("✅ 存入数据库成功！MVP 知识库构建完毕！🎉")
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