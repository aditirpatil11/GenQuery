import os
import streamlit as st
import openai

# 🧩 Disable Chroma telemetry early (critical fix)
import chromadb.telemetry.posthog as posthog
def _no_capture(*args, **kwargs): 
    pass
posthog.capture = _no_capture

from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document

# 🎬 Streamlit UI setup
st.set_page_config(page_title="GenQuery – AI Movie Assistant", page_icon="🎬", layout="centered")

st.markdown("""
# 🎬 **GenQuery – AI Movie Data Assistant**
Built with **LangChain • OpenAI • ChromaDB • Streamlit**

💡 *Try queries like:*
- Top 5 sci-fi movies after 2015  
- Who directed the most movies in 2020?  
- Average IMDb rating by genre
""")

# 🔑 Get OpenAI key
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Please add your `OPENAI_API_KEY` in Streamlit Secrets.")
    st.stop()
openai.api_key = api_key

# 🧠 Load IMDB data folder
rag_dir = os.path.join(os.path.dirname(__file__), "..", "rag_imdb")

# ⚙️ Use lightweight embedding model
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 🧩 Create or load Chroma DB (with telemetry disabled)
db = Chroma(persist_directory="./chroma_store", embedding_function=embeddings)

# 🗂️ If database is empty, add text files
if db._collection.count() == 0 and os.path.exists(rag_dir):
    for f in os.listdir(rag_dir):
        if f.endswith(".txt") or f.endswith(".jsonl"):
            with open(os.path.join(rag_dir, f), "r", encoding="utf-8") as file:
                text = file.read()
                db.add_documents([Document(page_content=text, metadata={"source": f})])
    db.persist()

# 🔍 Retriever setup
retriever = db.as_retriever(search_kwargs={"k": 3})

# 💬 LLM setup (using langchain_community.chat_models)
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

# 🧠 Retrieval QA Chain
prompt_template = """
You are an intelligent movie data assistant. 
Use the context below to answer the question accurately.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

# 🎯 Query box
query = st.text_input("🎯 Enter your question:")

if st.button("Run Query") and query:
    st.info("Running your query... please wait ⏳")
    try:
        response = qa_chain.invoke({"query": query})
        st.success(response["result"])
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

# Footer
st.markdown("---")
st.caption("Built with 💡 by Aditi | Powered by OpenAI • Streamlit • ChromaDB")
