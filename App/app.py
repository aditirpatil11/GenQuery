import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os

# Streamlit UI 
st.set_page_config(page_title="🎬 GenQuery", layout="wide")
st.title("🎬 GenQuery – AI Assistant")
st.caption("Powered by LangChain • OpenAI • FAISS • Streamlit")

st.markdown("""
💡 Example questions:
- Top 5 movies after 2020  
- Highest-rated movies by Christopher Nolan  
- Genre-wise average ratings
""")

openai_key = st.text_input("🔑 Enter your OpenAI API Key", type="password")
if not openai_key:
    st.warning("Please enter your OpenAI API key to start.")
    st.stop()
os.environ["OPENAI_API_KEY"] = openai_key

#  Load FAISS retriever 
try:
    st.info("Loading FAISS vector index...")
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("rag_imdb", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    st.success("✅ FAISS index loaded successfully!")
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

# Initialize LLM 
try:
    llm = ChatOpenAI(model="gpt-4-turbo", temperature=0.3)
except Exception as e:
    st.error(f"Model initialization error: {e}")
    st.stop()

#  Create retrieval chain 
prompt = ChatPromptTemplate.from_template("""
You are a movie expert. Use the context to answer clearly.
Question: {input}
Context: {context}
Answer:
""")

chain = create_stuff_documents_chain(llm, prompt)
retrieval_chain = create_retrieval_chain(retriever, chain)

#  Ask Query 
query = st.text_input("🎥 Ask a question about movies:")

if st.button("Ask"):
    if query.strip():
        with st.spinner("Thinking..."):
            try:
                result = retrieval_chain.invoke({"input": query})
                st.subheader("✅ Answer")
                st.write(result["answer"])
            except Exception as e:
                st.error(f"⚠️ Error while processing: {e}")
    else:
        st.warning("Please enter a question.")
