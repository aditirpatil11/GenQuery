import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval_qa.base import RetrievalQA

st.set_page_config(page_title="GenQuery – LLM SQL + RAG", page_icon="🧠", layout="wide")
st.title("🧠 GenQuery – AI Query & RAG Assistant")

st.markdown(
    """
    Ask any movie-related question in natural language.  
    GenQuery uses **RAG (Retrieval-Augmented Generation)**, **FAISS**, and **GPT-4o / LLaMA3**  
    to generate accurate SQL-like reasoning and natural answers.
    """
)

query = st.text_input("Enter your query:", placeholder="e.g., Show top 5 comedies after 2018")

# Load FAISS retriever
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = FAISS.load_local("rag_imdb", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"Error loading retriever: {e}")
    st.stop()

# Run LLM if query given
if query:
    with st.spinner("🔍 Generating intelligent answer..."):
        try:
            llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)
            qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
            answer = qa_chain.run(query)
            st.success("✅ Answer:")
            st.write(answer)
        except Exception as e:
            st.error(f"Error running query: {e}")

st.markdown("---")
st.caption("Built using Streamlit • LangChain • OpenAI • HuggingFace • FAISS")
