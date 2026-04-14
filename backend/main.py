import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import ZhipuAIEmbeddings
from zhipuai import ZhipuAI
os.environ["ZHIPUAI_API_KEY"] = ""

def process_pdf(file_path):
    print(f"开始处理文件: {file_path}")
    loader = PyMuPDFLoader(file_path)
    documents = loader.load()
    print(f"PDF 加载完成，共 {len(documents)} 页。")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )#切分
    chunked_docs = text_splitter.split_documents(documents)
    embeddings = ZhipuAIEmbeddings(
        api_key="os.environ['ZHIPUAI_API_KEY']",
        model="embedding-3"  # 智谱官方推荐的向量大模型名称
    )
    print(f"文本切块完成，一共切出了 {len(chunked_docs)} 个文本块！")
    print("first chunk metadata:", chunked_docs[0].metadata)
    print("first chunk preview:\n", chunked_docs[0].page_content[:300])
    vectorstore = Chroma.from_documents(
        documents=chunked_docs,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    print("存入数据库成功！MVP 知识库构建完毕！🎉")
    return True
