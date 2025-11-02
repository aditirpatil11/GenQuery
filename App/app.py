import os
import streamlit as st
import openai
import chromadb

# Page setup
st.set_page_config(
    page_title="GenQuery – AI Assistant",
    page_icon="🎥",
    layout="centered"
)

# App header
st.markdown("""
# 🎬 **GenQuery – AI Assistant**
Built with **LangChain • OpenAI • ChromaDB • Streamlit**

💡 *Try queries like:*
- Top 5 sci-fi movies after 2015  
- Who directed the most movies in 2020?  
- Average IMDb rating by genre
""")

# Load API key
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Please add your `OPENAI_API_KEY` in Streamlit Secrets.")
    st.stop()

# ✅ Initialize OpenAI (old-style to fix proxy error)
openai.api_key = api_key

# 🧩 Initialize Chroma vector store
chroma_client = chromadb.Client()
collection = chroma_client.create_collection("imdb_data")

# Load text docs from rag_imdb folder
rag_dir = os.path.join(os.path.dirname(__file__), "..", "rag_imdb")
if os.path.exists(rag_dir):
    for f in os.listdir(rag_dir):
        if f.endswith(".txt") or f.endswith(".jsonl"):
            with open(os.path.join(rag_dir, f), "r", encoding="utf-8") as file:
                text = file.read()
                collection.add(documents=[text], ids=[f])

# Query input
query = st.text_input("🎯 Enter your question:")
if st.button("Run Query") and query:
    st.info("Running your query... please wait ⏳")

    # Retrieve top documents
    results = collection.query(query_texts=[query], n_results=3)
    context = " ".join(results["documents"][0]) if results["documents"] else ""

    prompt = f"""Use the following context to answer the question.

    Context:
    {context}

    Question: {query}
    """

    # Ask LLM
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{"role": "user", "content": prompt}],
        )
        st.success(response.choices[0].message.content)
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# Footer
st.markdown("---")
st.caption("Powered by OpenAI • LangChain • Streamlit • ChromaDB")
