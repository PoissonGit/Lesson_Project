# RAG.py
import streamlit as st
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

# ------------------------------
# 页面和会话配置
# ------------------------------
st.set_page_config(page_title="医学问答助手", layout="wide")  
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'step' not in st.session_state:
    st.session_state.step = 'ask'
if 'stored_docs' not in st.session_state:
    st.session_state.stored_docs = []
if 'prev_query' not in st.session_state:
    st.session_state.prev_query = ''

# ------------------------------
# 基础组件初始化
# ------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
llm = ChatOpenAI(
    model_name="deepseek-chat",
    temperature=0,
    openai_api_key="sk-7386338a6e5b41c7ba359c998f4a5fbc",
    openai_api_base="https://api.deepseek.com/v1"
)
embedding_model = HuggingFaceBgeEmbeddings(
    model_name="BAAI/bge-m3",
    model_kwargs={"device": device},
    encode_kwargs={"normalize_embeddings": True},
    query_instruction=""
)
vector_store = Milvus(
    collection_name="medical_docs",
    embedding_function=embedding_model,
    connection_args={"host": "localhost", "port": "19530"}
)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})

# 重排模型
reranker_tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-reranker-base")
reranker_model = AutoModelForSequenceClassification.from_pretrained("BAAI/bge-reranker-base")
reranker_model.to(device)
def rerank_documents(query, docs):
    reranked = []
    for doc in docs:
        inputs = reranker_tokenizer(query, doc.page_content, return_tensors="pt", truncation=True).to(device)
        with torch.no_grad(): score = reranker_model(**inputs).logits.squeeze().item()
        reranked.append((score, doc))
    reranked.sort(key=lambda x: x[0], reverse=True)
    return [d for _, d in reranked[:10]]

# 文本清洗与切分（见前省略实现，保持同原）
def clean_html(html_content: str) -> str:
    soup = BeautifulSoup(html_content, "html.parser")
    for elem in soup(["script", "style"]): elem.extract()
    text = soup.get_text(separator="\n")
    return re.sub(r'\n+', '\n', text).strip()
def clean_markdown(md_content: str) -> str:
    text = re.sub(r'```.*?```', '', md_content, flags=re.DOTALL)
    text = re.sub(r'`([^`]+)`', r'\1', text)
    text = re.sub(r'!\[.*?\]\(.*?\)', '', text)
    text = re.sub(r'\[(.*?)\]\(.*?\)', r'\1', text)
    text = re.sub(r'^#{1,6}\s*', '', text, flags=re.MULTILINE)
    return re.sub(r'\n+', '\n', text).strip()

def load_and_split_documents(folder_path):
    docs=[]; spl=RecursiveCharacterTextSplitter(separators=["\n\n","。","！"],chunk_size=500,chunk_overlap=50,length_function=len)
    for f in os.listdir(folder_path):
        p=os.path.join(folder_path,f)
        if f.endswith((".html",".md")):
            raw=open(p,encoding='utf-8').read()
            text = clean_html(raw) if f.endswith(".html") else clean_markdown(raw)
            for i,ch in enumerate(spl.split_text(text)): docs.append((ch,{"source":f,"chunk":i}))
    return docs

# ------------------------------
# UI 渲染
# ------------------------------
st.title("🩺 医学知识问答助手")

# 用户提问阶段
if st.session_state.step == 'ask':
    # 如果有前置问题，显示为上下文
    if st.session_state.prev_query:
        st.info(f"📌 上一话题：{st.session_state.prev_query}")
    query = st.text_input("请输入您的问题：", key="input_query")
    if query:
        # 若存在 prev_query，则迭代拼接
        full_query = (st.session_state.prev_query + ' ' + query).strip() if st.session_state.prev_query else query
        st.session_state.chat_history.append(("用户", query))
        # 更新 prev_query 为 full_query
        st.session_state.prev_query = full_query
        st.write("🔍 正在检索并重排文档...")
        docs = retriever.get_relevant_documents(full_query)
        reranked = rerank_documents(full_query, docs)
        st.session_state.stored_docs = reranked
        st.session_state.step = 'review'
        st.rerun()

# 文档审阅阶段
elif st.session_state.step == 'review':
    st.write("📄 以下是检索到的文档片段，请选择进一步使用的内容：")
    selections = []
    with st.form("review_form"):
        for idx, doc in enumerate(st.session_state.stored_docs):
            sel = st.checkbox(f"文档 {idx+1} 来源:{doc.metadata['source']}…", key=f"sel{idx}")
            st.write(doc.page_content[:200] + "…")
            if sel:
                selections.append(doc.page_content)
        submitted = st.form_submit_button("✅ 使用选中内容生成答案")
    if submitted:
        st.session_state.selected_context = (
            selections if selections else [d.page_content for d in st.session_state.stored_docs]
        )
        st.session_state.step = 'answer'
        st.rerun()
    if st.button("🔄 修改查询内容"):
        st.session_state.step = 'ask'
        st.rerun()

# 回答生成阶段
elif st.session_state.step == 'answer':
    context = "\n\n".join(st.session_state.selected_context)
    user_msg = st.session_state.chat_history[-1][1]
    prompt = f"你是一位医学专家，请根据以下资料回答：\n{context}\n问题：{user_msg}\n详细回答："
    st.write("✍️ 正在生成回答...")
    ans = llm.predict(prompt)
    st.session_state.chat_history.append(("助手", ans))
    st.write(f"**回答**：{ans}")
    # “追问”跳回ask，保留prev_query用于迭代
    if st.button("💡 追问"):
        st.session_state.step = 'ask'
        st.rerun()
    # 新问题时重置上下文
    if st.button("🏠 新问题"): 
        st.session_state.step = 'ask'
        st.session_state.prev_query = ''
        st.session_state.stored_docs = []
        st.rerun()

# 显示多轮历史（可折叠）
st.markdown("---")
with st.expander("📜 查看对话历史", expanded=False):
    for role, msg in st.session_state.chat_history:
        st.chat_message(role).markdown(msg)
