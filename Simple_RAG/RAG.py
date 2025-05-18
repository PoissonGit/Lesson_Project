#RAG.py
import streamlit as st
st.set_page_config(page_title="医学问答助手", layout="wide")
from langchain_community.chat_models import ChatOpenAI
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import Milvus
from langchain.memory import ConversationBufferMemory
from bs4 import BeautifulSoup
import re
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch


device = "cuda" if torch.cuda.is_available() else "cpu"

# ------------------------------
# 文档清洗函数
# ------------------------------
def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for elem in soup(["script", "style"]):
        elem.extract()
    text = soup.get_text(separator="\n")
    text = re.sub(r'\s+\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

def clean_markdown(md_content: str) -> str:
    text = re.sub(r'```.*?```', '', md_content, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    text = re.sub(r'\r\n', '\n', text)
    text = re.sub(r'\n+', '\n', text)
    return text.strip()

# ------------------------------
# 加载与切分本地文档
# ------------------------------
def load_and_split_documents(folder_path):
    documents = []
    splitter = RecursiveCharacterTextSplitter(
        separators=["\n\n", "。", "．", ".", "！", "!", "？", "?", "，", "、", ","],
        chunk_size=500,
        chunk_overlap=50,
        length_function=len
    )
    for file in os.listdir(folder_path):
        path = os.path.join(folder_path, file)
        if file.endswith(".html"):
            with open(path, 'r', encoding='utf-8') as f:
                raw = f.read()
                text = clean_html(raw)
        elif file.endswith(".md"):
            with open(path, 'r', encoding='utf-8') as f:
                raw = f.read()
                text = clean_markdown(raw)
        else:
            continue
        chunks = splitter.split_text(text)
        for i, chunk in enumerate(chunks):
            documents.append((chunk, {"source": file, "chunk": i}))
    return documents

# ------------------------------
# 初始化 LLM 和 Embedding 模型
# ------------------------------
llm = ChatOpenAI(
    model_name="deepseek-chat",  # DeepSeek模型名称
    temperature=0,
    openai_api_key="DS_KEY",  # 需要可用的DeepSeek API Key
    openai_api_base="https://api.deepseek.com/v1"  # DeepSeek API地址
)

embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
    query_instruction=""
)


# ------------------------------
# 使用重排模型 BGE-reranker
# ------------------------------
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
reranker_model.to(device)

def rerank_documents(query, docs):
    reranked = []
    for doc in docs:
        inputs = reranker_tokenizer(query, doc.page_content, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad():
            score = reranker_model(**inputs).logits.squeeze().item()
        reranked.append((score, doc))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [doc for _, doc in reranked[:3]]

# ------------------------------
# 构建知识库（首次运行）
# ------------------------------
# docs = load_and_split_documents("./md")
# texts = [d[0] for d in docs]
# metadatas = [d[1] for d in docs]
# st.write("📦 正在初始化向量存储...")
# vector_store = Milvus.from_texts(
#     texts=texts,
#     embedding=embedding_model,
#     metadatas=metadatas,
#     connection_args={"host": "localhost", "port": "19530"},
#     collection_name="medical_docs"
# )

# ------------------------------
# 连接向量数据库 Milvus
# ------------------------------
vector_store = Milvus(
    collection_name="medical_docs",
    embedding_function=embedding_model,
    connection_args={"host": "localhost", "port": "19530"}
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# ------------------------------
# 构建带记忆的对话链
# ------------------------------
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
qa_chain = ConversationalRetrievalChain.from_llm(
    llm=llm,
    retriever=retriever,
    memory=memory
)

# ------------------------------
# Streamlit 页面布局
# ------------------------------
st.title("🩺 医学知识问答助手")
# ✅ 输入与回答逻辑
user_question = st.text_input("请输入您的问题：", key="input")

if user_question:
    st.write("正在检索文档...")
    docs = retriever.get_relevant_documents(user_question)

    st.write("正在重排序文档...")
    reranked_docs = rerank_documents(user_question, docs)

    context = "\n\n".join([doc.page_content for doc in reranked_docs])
    prompt = f"""你是一位医学专家助理，请根据以下资料回答用户的问题：\n{context}\n\n问题：{user_question}\n请根据资料详细作答："""

    st.write("✍️ 正在生成回答...")
    answer = llm.predict(prompt)
    st.write(f"答案：\n{answer}")

