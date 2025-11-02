import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA

st.set_page_config(page_title="GenQuery – LLM SQL Assistant", page_icon="🧠", layout="wide")
st.title("🧠 GenQuery – LLM-Based SQL Interpreter")

st.markdown(
    "Ask questions in plain English. The app uses **RAG**, **FAISS**, and **OpenAI/LLaMA-3** "
    "to generate accurate SQL queries and explain results interactively."
)

query = st.text_input("Enter your natural-language query:", placeholder="e.g., Show top 5 action movies after 2015")

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = FAISS.load_local("rag_imdb", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

if query:
    with st.spinner("🔍 Thinking... generating SQL and fetching results..."):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            response = qa_chain.run(query)
            st.subheader("💡 LLM Response")
            st.write(response)
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built using Streamlit, LangChain, and OpenAI/LLaMA-3.")

