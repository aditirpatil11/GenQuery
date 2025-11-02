import os
import streamlit as st
from openai import OpenAI
import chromadb


# Load OpenAI API key

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🚨 Missing OpenAI API key. Please add it under 'Secrets' in Streamlit Cloud.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# Initialize ChromaDB (lightweight vector store)

chroma_client = chromadb.Client()
collection = chromadb.Client().create_collection("imdb_docs")

docs_dir = os.path.join(os.path.dirname(__file__), "..", "rag_imdb")

# Load any .txt or .jsonl files as context
if os.path.exists(docs_dir):
    for filename in os.listdir(docs_dir):
        if filename.endswith(".txt"):
            with open(os.path.join(docs_dir, filename), "r", encoding="utf-8") as f:
                content = f.read()
                collection.add(documents=[content], ids=[filename])
else:
    st.warning("⚠️ No RAG data found — please add .txt files to the rag_imdb folder!")


# Streamlit UI

st.set_page_config(page_title="GenQuery – AI Assistant", page_icon="🎬", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #F5C518;'>🎬 GenQuery -  AI Assistant </h1>
    <h3 style='text-align: center; color: #bbb;'>Built with <b>LLM + RAG + Streamlit</b></h3>
    <p style='text-align: center; color: #888;'>Ask natural questions about your IMDb data or text files.</p>
    """,
    unsafe_allow_html=True
)
st.divider()

query = st.text_input("🔍 Ask a question:")


# Run RAG Query

if query:
    st.info("Running query... please wait ⏳")

    # Retrieve relevant documents
    results = collection.query(query_texts=[query], n_results=3)
    context = " ".join([doc for doc in results["documents"][0]]) if results["documents"] else ""

    # Build LLM prompt
    prompt = f"""
    You are an intelligent movie assistant. 
    Use the following context to answer the question clearly and concisely.

    Context:
    {context}

    Question:
    {query}

    Answer:
    """

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.5
        )
        st.success(response.choices[0].message.content.strip())

        if context:
            with st.expander("🧾 Sources used"):
                st.text(context[:1000] + "...")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.markdown("💡 Try: *'List top directors from the database'* or *'Summarize IMDb notes from text files.'*")

