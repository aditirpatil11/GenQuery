import streamlit as st
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
import os

# PAGE CONFIG 
st.set_page_config(page_title="GenQuery – AI Movie RAG Assistant", page_icon="🎬", layout="wide")

st.markdown("""
# 🎬 GenQuery – AI Movie RAG Assistant

Ask questions like:
- "List top 5 movies after 2020"
- "Visualize number of movies per genre"
- "Which directors have the highest average rating?"
""")

#  LOAD RETRIEVER 
retriever = None
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    if os.path.exists("rag_imdb/faiss_index.bin") or os.path.exists("rag_imdb/index.faiss"):
        db = FAISS.load_local("rag_imdb", embeddings, allow_dangerous_deserialization=True)
        retriever = db.as_retriever()
        st.success("FAISS retriever loaded successfully!")
    else:
        st.warning("FAISS index not found — running in LLM-only mode.")
except Exception as e:
    st.warning(f"Could not load retriever: {e}")
    retriever = None

#  LLM SETUP 
llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

# If retriever exists, use RetrievalQA; else fallback to plain LLM
if retriever:
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, return_source_documents=True)
else:
    qa_chain = None

#  QUERY INPUT 
st.subheader("Ask your question:")
user_query = st.text_input("e.g., Top rated sci-fi movies after 2015")

if user_query:
    with st.spinner("Thinking..."):
        try:
            if qa_chain:
                result = qa_chain.invoke({"query": user_query})
                st.markdown("### 💡 Answer:")
                st.write(result["result"])
            else:
                response = llm.invoke(user_query)
                st.markdown("### 💡 Answer:")
                st.write(response.content)
        except Exception as e:
            st.error(f"Error: {e}")


