import os
from pathlib import Path
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings

# 请确保在环境变量或���里配置好 API KEY
os.environ["ZHIPUAI_API_KEY"] = "你的智谱API_KEY"


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