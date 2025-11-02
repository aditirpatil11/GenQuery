import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
import os

# Load API key automatically from Streamlit Secrets
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error(" Missing OpenAI API key. Please add it in Streamlit Secrets.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

# Page setup 
st.set_page_config(page_title="GenQuery – AI Assistant", page_icon="🎬", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #F5C518;'>🎬 GenQuery – AI Movie Data Assistant</h1>
    <p style='text-align: center; color: #BBBBBB;'>Built with <b>LangChain</b> • <b>OpenAI</b> • <b>FAISS</b> • <b>Streamlit</b></p>
    """,
    unsafe_allow_html=True,
)

st.markdown("---")

st.subheader("🎥 Try queries like:")
st.markdown("""
- *Top 5 sci-fi movies after 2015*  
- *Who directed the most movies in 2020?*  
- *Average IMDb rating by genre*
""")

query = st.text_input("🔍 Enter your question:")

if query:
    st.info("Running your query... please wait ⏳")

    # Example LLM usage (you can customize this for FAISS retrieval later)
    llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
    result = llm.invoke(query)

    st.success(result.content)

