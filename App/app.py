import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

st.set_page_config(page_title="GenQuery – AI Movie Assistant", page_icon="🎬", layout="wide")
st.title("🎬 GenQuery – AI Movie RAG Assistant")

st.markdown("""
Ask questions like:
- "List top 5 movies after 2020"
- "Visualize number of movies per genre"
- "Which directors have the highest average rating?"
""")

query = st.text_input("Enter your question:", placeholder="e.g., Top rated sci-fi movies after 2015")

# Load FAISS retriever
try:
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    retriever = FAISS.load_local("rag_imdb", embeddings, allow_dangerous_deserialization=True)
except Exception as e:
    st.error(f"Error loading retriever: {e}")
    st.stop()

# Build LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Prompt template for RAG
prompt = ChatPromptTemplate.from_template("""
You are a movie data analyst with access to IMDb-style data.
Use the provided context to answer the user's question accurately.

Context:
{context}

Question:
{question}

Answer in a clear and concise way.
""")

# Create RAG pipeline
combine_chain = create_stuff_documents_chain(llm, prompt)
rag_chain = create_retrieval_chain(retriever, combine_chain)

# Run on query
if query:
    with st.spinner("🔍 Thinking... generating intelligent response..."):
        try:
            response = rag_chain.invoke({"question": query})
            st.success("✅ Answer:")
            st.write(response["answer"])
        except Exception as e:
            st.error(f"Error: {e}")

st.markdown("---")
st.caption("Built using Streamlit • LangChain • OpenAI • HuggingFace • FAISS")
