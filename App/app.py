import os
import streamlit as st
from openai import OpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document


# Load OpenAI key from Streamlit secrets

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🚨 Missing OpenAI API key. Please add it under 'Secrets' in Streamlit Cloud.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# Build FAISS vector store (in-memory)

st.set_page_config(page_title="GenQuery – AI Assistant", page_icon="🎬", layout="centered")

st.markdown(
    """
    <h1 style='text-align: center; color: #F5C518;'>🎬 GenQuery – AI Assistant </h1>
    <h3 style='text-align: center; color: #bbb;'>Built with <b>LLM + RAG + Streamlit</b></h3>
    <p style='text-align: center; color: #888;'>Ask natural questions about your IMDb data or text files.</p>
    """,
    unsafe_allow_html=True
)

docs_dir = os.path.join(os.path.dirname(__file__), "..", "rag_imdb")
text_data = ""

# Load text files
if os.path.exists(docs_dir):
    for file in os.listdir(docs_dir):
        if file.endswith(".txt"):
            with open(os.path.join(docs_dir, file), "r", encoding="utf-8") as f:
                text_data += f.read() + "\n"
else:
    st.warning("⚠️ No RAG documents found. Add .txt files in 'rag_imdb/'.")

# Split text into chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
chunks = splitter.split_text(text_data)

# Create embeddings
if chunks:
    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
    db = FAISS.from_texts(chunks, embeddings)
else:
    db = None


# Query interface

query = st.text_input("🔍 Ask a question about your data:")

if query:
    st.info("Running RAG query... please wait ⏳")

    # Retrieve relevant chunks
    if db:
        docs = db.similarity_search(query, k=3)
        context = "\n".join([d.page_content for d in docs])
    else:
        context = "No context available."

    # Construct prompt
    prompt = f"""
    You are a helpful movie assistant. Use the context below to answer the user's question.
    
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
            temperature=0.5,
        )
        st.success(response.choices[0].message.content.strip())

        if db:
            with st.expander("🧾 Context used"):
                st.text(context[:1000] + "...")
    except Exception as e:
        st.error(f"⚠️ Error: {e}")
else:
    st.markdown("💡 Try: *'Top sci-fi movies after 2015'* or *'Who directed the most films in 2020?'*")
