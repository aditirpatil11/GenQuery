import os
import streamlit as st
from openai import OpenAI
import chromadb

st.set_page_config(page_title="GenQuery – LLM + RAG + Streamlit", page_icon="🎬")

# Load key
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("Please add your OPENAI_API_KEY in Streamlit secrets.")
    st.stop()
client = OpenAI(api_key=api_key)

# Init Chroma (vector store)
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("imdb")

# Load text files for context
docs_dir = os.path.join(os.path.dirname(__file__), "..", "rag_imdb")
for f in os.listdir(docs_dir):
    if f.endswith(".txt"):
        with open(os.path.join(docs_dir, f), "r", encoding="utf-8") as file:
            text = file.read()
            collection.add(documents=[text], ids=[f])

# UI
st.markdown("## 🎬 GenQuery – AI Assistant")
st.write("Ask movie-related questions using RAG + OpenAI")

q = st.text_input("Enter your question:")
if q:
    st.info("Thinking...")
    results = collection.query(query_texts=[q], n_results=3)
    context = " ".join(results["documents"][0])
    prompt = f"Answer based on this context:\n{context}\n\nQuestion: {q}"

    resp = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}],
    )
    st.success(resp.choices[0].message.content)
