import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains.retrieval import create_retrieval_chain
import os

st.set_page_config(page_title="🎬 GenQuery", layout="centered")
st.title("🎬 GenQuery – AI Movie Data Assistant")
st.caption("Built with LangChain • OpenAI • FAISS • Streamlit")

st.markdown("""
**Try queries like:**
- “Top 5 sci-fi movies after 2015”
- “Who directed the most movies in 2020?”
- “Average IMDb rating by genre”
""")

api_key = st.text_input("🔑 Enter your OpenAI API key", type="password")
if not api_key:
    st.warning("Please enter your OpenAI API key to continue.")
    st.stop()

os.environ["OPENAI_API_KEY"] = api_key

try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    db = FAISS.load_local("rag_imdb", embeddings, allow_dangerous_deserialization=True)
    retriever = db.as_retriever()
    st.success("✅ FAISS index loaded!")
except Exception as e:
    st.error(f"Error loading FAISS index: {e}")
    st.stop()

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.2)

prompt = ChatPromptTemplate.from_template("""
You are a movie expert. Use the provided context to answer clearly.
Question: {input}
Context: {context}
Answer:
""")

chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, chain)

query = st.text_input("🎥 Ask a question about movies:")

if st.button("Search"):
    if query.strip():
        with st.spinner("Generating answer..."):
            try:
                response = rag_chain.invoke({"input": query})
                st.markdown("### 🎯 Answer:")
                st.write(response["answer"])
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        st.warning("Please enter a question first.")

