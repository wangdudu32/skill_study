from pathlib import Path
from dotenv import load_dotenv
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_openai import ChatOpenAI,OpenAIEmbeddings
from langchain_classic.chains import RetrievalQA
from langchain_core.prompts import PromptTemplate
from langchain_community.document_loaders import TextLoader,PyPDFLoader,Docx2txtLoader

def load_documents_from_directory(directory: str) -> list[Document]:
    # 加载目录以及子目录中的 TXT MD PDF DOCX 文件
    root = Path(directory).expanduser() # 将目录中的 ~ 展开成绝对路径
    if not root.is_dir():
        raise ValueError(f"目录不存在/不是目录:{root}")

    documents:list[Document]=[]

    # rglob(*)：遍历目录以及所有子目录
    for file_path in sorted(root.rglob("*")):
        if not file_path.is_file():     # 如果不是文件，则跳过
            continue    
        
        suffix = file_path.suffix.lower()   # 获取文件名后缀并转换成小写

        # 根据文件类型选择加载器
        if suffix in {".txt",".md"}:
            loader = TextLoader(str(file_path),encoding = "utf-8")
        elif suffix == ".pdf":
            loader = PyPDFLoader(str(file_path),mode="page")
        elif suffix == ".docx":
            loader = Docx2txtLoader(str(file_path))
        else:
            print(f"跳过不支持的文件:{file_path}")
            continue
        
        documents.extend(loader.load())

    print(f"已加载 {len(documents)} 个 Document 对象")
    return documents

def split_documents(documents:list[Document],chunk_size:int = 500, chunk_overlap:int=50)->list[Document]:
    # 切分文档
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = chunk_size,
        chunk_overlap = chunk_overlap,
        length_function = len,
        separators = ["\n\n","\n", "。", "，", "！","？",""]
    )
    splits = text_splitter.split_documents(documents)
    print(f"已将 {len(documents)} 个文档切分成 {len(splits)} 个文本块")
    return splits

def create_vector_store(splits:list[Document],persist_directory:str="./chroma_db")->Chroma:
    # 创建向量数据库
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma.from_documents(
        documents = splits,
        embedding = embeddings,
        persist_directory=persist_directory
    )
    print(f"向量数据库已创建，共 {vector_store._collection.count()} 条记录")
    return vector_store

def load_vector_store(persist_directory:str="./chroma_db")->Chroma:
    # 加载已有的向量数据库
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    vector_store = Chroma(
        persist_directory=persist_directory,
        embedding_function=embeddings
    )
    print(f"向量数据库已加载，共 {vector_store._collection.count()} 条记录")
    return vector_store

def create_rag_chain(vector_store:Chroma):
    # 创建 RAG 问答链
    # 自定义 Prompt 模板
    template = """ 你是一个专业的客服助手。请基于以下上下文回答用户的问题。如果上下文中没有相关信息，请说'抱歉，我没找到相关信息'。回答要简洁明了，使用中文。
    
上下文：{context}

用户问题：{question}

回答："""

    prompt = PromptTemplate(template = template,input_variables = ["context","question"])
    llm = ChatOpenAI(model = "gpt-4o-mini",temperature=0)

    # 创建检索链
    rag_chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type = "stuff",
        retriever = vector_store.as_retriever(search_kwargs={"k":3}),
        chain_type_kwargs = {"prompt":prompt},
        return_source_documents = True
    )

    return rag_chain

def main():
    # 主函数
    # 使用脚本所在目录，避免从其他工作目录启动时找不到文件
    base_dir = Path(__file__).resolve().parent
    db_path = base_dir / "chroma_db"
    # 加载环境变量
    load_dotenv(base_dir / ".env")

    # 检查向量数据库是否已存在
    db_exists = db_path.is_dir()

    if db_exists:
        print("发现已存在的数据库，正在加载...")
        vector_store = load_vector_store(str(db_path))
    else:
        print("创建新的数据库...")
        # 1.加载文档
        documents = load_documents_from_directory(str(base_dir / "documents"))
        # 2.切分文档
        splits = split_documents(documents)
        if not splits:
            raise ValueError("文档目录中没有可用于问答的文本内容")
        # 3.创建向量数据库
        vector_store = create_vector_store(splits, str(db_path))

    # 4.创建 RAG 链
    rag_chain = create_rag_chain(vector_store)

    # 测试查询
    print(f"{'='*100}")
    while True:
        question = input("问题（输入 q 退出）:").strip()
        if question.lower() == "q":
            break
        if not question:
            continue
        print("="*60)

        result = rag_chain.invoke({"query":question})
        answer = result['result']
        sources = result['source_documents']

        print(f"回答：{answer}")
        print('\n参考来源：')
        for i,doc in enumerate(sources,1):
            source = doc.metadata.get("source", "未知来源")
            page = doc.metadata.get("page")
            # PDF 页码从 0 开始；TXT、MD、DOCX 通常没有 page 字段
            page_info = f" (第 {page + 1} 页)" if page is not None else ""
            print(f"\t[{i}] {source}{page_info}")
            print(f"\t\t片段：{doc.page_content[:50]}...")
        
        print("\n" + "="*100 + "\n")

if __name__ == "__main__":
    main()
