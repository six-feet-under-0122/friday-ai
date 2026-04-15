from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader


# ----------！！！上传！！！----------
def load_pdf_pages(pdf_path: str):
    # 1) 把用户传入的路径转换为标准绝对路径
    pdf_path = Path(pdf_path).expanduser().resolve()

    # 2) 先做存在性检查，错误信息更明确
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found: {pdf_path}")

    # 3) 用 LangChain loader 加载
    loader = PyMuPDFLoader(str(pdf_path))
    docs = loader.load()  # List[Document]一页一个document

    # 4) 可选：再做一次“至少读到一页”的断言
    # 如果docs为空列表
    if not docs:
        raise ValueError(f"Loaded 0 pages from PDF: {pdf_path}")

    return docs
docs=load_pdf_pages("D:\\team\\friday-ai\\backend\\test.pdf")
print(docs[0].metadata)

# ----------！！！切分！！！----------
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

def chunk_documents(
    page_docs,
    chunk_size: int = 800,
    chunk_overlap: int = 120,
):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],  # 中文/英文都兼顾 有先后优先顺序
    )

    chunks = []
    for d in page_docs:
        # d 是一页的 Document
        page_text = (d.page_content or "").strip()
        if not page_text:
            continue  # 空页跳过

        # split_text 得到 List[str]，内元素为切片文本
        pieces = splitter.split_text(page_text)

        # 把每个 piece 再包装成 Document，并继承原 metadata
        for j, piece in enumerate(pieces):
            md = dict(d.metadata or {})
            md["chunk_id"] = f"p{md.get('page','?')}_c{j}"
            md["chunk_size"] = len(piece)
            chunks.append(Document(page_content=piece, metadata=md))

    return chunks
docs = load_pdf_pages(r"D:\team\friday-ai\backend\test.pdf")
chunks = chunk_documents(docs, chunk_size=800, chunk_overlap=120)

import os
from dotenv import load_dotenv
load_dotenv()
from langchain_community.vectorstores import Chroma

# 下面两个 import 可能因版本不同需调整
from langchain_community.embeddings import ZhipuAIEmbeddings

PERSIST_DIR = "chroma_db"
COLLECTION_NAME = "pdf_chunks"

def get_embeddings():


    return ZhipuAIEmbeddings(
        api_key="2047124c867446008485c8ff86bda1d1.kKkhKxIoC34ZSSBI",                 # 有些版本参数名可能叫 zhipuai_api_key
        model="embedding-3",
    )

def get_or_build_vectorstore(chunks):
    embeddings = get_embeddings()

    if Path(PERSIST_DIR).exists():
        # 复用已有库
        vs = Chroma(
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
            embedding_function=embeddings,
        )
        # 可选：如果你发现 count==0，说明目录存在但空
        if vs._collection.count() == 0:
            vs.add_documents(chunks)
            vs.persist()
    else:
        # 首次建库
        vs = Chroma.from_documents(
            documents=chunks,
            embedding=embeddings,
            persist_directory=PERSIST_DIR,
            collection_name=COLLECTION_NAME,
        )
        vs.persist()

    return vs



print("页数总共有:", len(docs))
print("一共切成:", len(chunks),"块")
print("="*50)
print("第一个chunk metadata:", chunks[0].metadata)
print("第一个chunk preview:\n", chunks[0].page_content)
vs = get_or_build_vectorstore(chunks)
print("Chroma count:", vs._collection.count())