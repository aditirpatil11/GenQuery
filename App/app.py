import streamlit as st
import os
from datetime import datetime
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
from langchain.prompts import ChatPromptTemplate

# -------------------- Page Config --------------------
st.set_page_config(page_title="🎬 GenQuery – AI SQL & Movie Assistant", layout="wide")

# -------------------- Custom CSS --------------------
st.markdown("""
<style>
body, .main { background-color: #0e1117; color: white; }
h1, h2, h3 { color: #f5c518; }
.sidebar .sidebar-content { background-color: #161a21; }
.stButton>button { background-color: #f5c518; color: black; border-radius: 8px; }
.chat-box {
    border-radius: 8px;
    background-color: #1b1f27;
    padding: 1rem;
    margin-top: 0.5rem;
}
</style>
""", unsafe_allow_html=True)

# -------------------- Header --------------------
st.title("🎬 GenQuery – AI SQL & Movie RAG Assistant")
st.caption("Built with LangChain • FAISS • OpenAI • Streamlit")

st.markdown("""
Ask any data-driven movie question:  
🔹 “List top 5 movies after 2020”  
🔹 “Which directors have the highest average rating?”  
🔹 “Show number of movies per genre after 2015”
""")

# -------------------- Sidebar --------------------
with st.sidebar:
    st.header("⚙️ Settings")
    model_choice = st.selectbox("Choose Model", ["OpenAI GPT-4", "LLaMA 3 (local)"])
    openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
    theme_choice = st.radio("Theme", ["Dark", "Light"], index=0)
    st.divider()
    st.caption("👩‍💻 Powered by LangChain + FAISS + Streamlit")

# -------------------- Apply Theme --------------------
if theme_choice == "Light":
    st.markdown("<style>body, .main { background-color: white; color: black; }</style>", unsafe_allow_html=True)

# -------------------- Initialize Session State --------------------
if "history" not in st.session_state:
    st.session_state.history = []

# -------------------- Load FAISS --------------------
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("rag_imdb", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
except Exception as e:
    st.error(f"⚠️ Could not load FAISS index: {e}")
    st.stop()

# -------------------- LLM Setup --------------------
try:
    if model_choice.startswith("OpenAI"):
        if not openai_key:
            st.warning("Please enter your OpenAI API key to continue.")
            st.stop()
        os.environ["OPENAI_API_KEY"] = openai_key
        llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
    else:
        llm = ChatOpenAI(model="meta-llama/Llama-3-8b", temperature=0.3)
except Exception as e:
    st.error(f"❌ Model initialization failed: {e}")
    st.stop()

# -------------------- Create Chain --------------------
prompt = ChatPromptTemplate.from_template("""
You are a movie data expert. Use the retrieved context to answer accurately.

Question: {input}
Context: {context}
Answer:
""")
chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, chain)

# -------------------- User Query --------------------
query = st.text_input("💬 Ask your movie-related question:")

if st.button("Ask AI"):
    if query.strip():
        with st.spinner("🤔 Thinking..."):
            try:
                result = retrieval_chain.invoke({"input": query})
                answer = result["answer"].strip()
                timestamp = datetime.now().strftime("%H:%M:%S")
                st.session_state.history.append((timestamp, query, answer))

                st.markdown("### ✅ Answer")
                st.markdown(f"<div class='chat-box'>{answer}</div>", unsafe_allow_html=True)
            except Exception as e:
                st.error(f"⚠️ Error: {e}")
    else:
        st.warning("Please enter a question before submitting.")

# -------------------- Chat History --------------------
if st.session_state.history:
    st.markdown("### 🕘 Chat History")
    for t, q, a in reversed(st.session_state.history[-5:]):
        st.markdown(f"**🕒 {t}**  \n💭 *{q}*  \n📘 **{a}**")
