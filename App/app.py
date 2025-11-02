import os
import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
import pickle

# 🎬 Page setup
st.set_page_config(page_title="GenQuery – AI RAG Assistant", page_icon="🎬", layout="centered")

st.markdown("""
# 🎬 **GenQuery – AI Knowledge Assistant**
Built with **LangChain • OpenAI • FAISS • Streamlit**

💡 *Ask questions like:*
- "List top sci-fi movies after 2015"  
- "What was the highest-grossing film in 2020?"  
- "Summarize the director trends in the dataset."
""")

# 🔑 API key
api_key = st.secrets.get("OPENAI_API_KEY")
if not api_key:
    st.error("⚠️ Please add your `OPENAI_API_KEY` in Streamlit Secrets.")
    st.stop()

# 🧠 Prepare embeddings
embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# 🗂️ Load or create FAISS index
if os.path.exists("faiss_index.pkl"):
    with open("faiss_index.pkl", "rb") as f:
        db = pickle.load(f)
else:
    # Folder where your RAG text files are stored
    rag_dir = os.path.join(os.path.dirname(__file__), "..", "rag_imdb")
    docs = []
    if os.path.exists(rag_dir):
        for f in os.listdir(rag_dir):
            if f.endswith(".txt") or f.endswith(".jsonl"):
                with open(os.path.join(rag_dir, f), "r", encoding="utf-8") as file:
                    text = file.read()
                    docs.append(Document(page_content=text, metadata={"source": f}))
    db = FAISS.from_documents(docs, embeddings)
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(db, f)

retriever = db.as_retriever(search_kwargs={"k": 3})

# 💬 LLM setup
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, openai_api_key=api_key)

# 🧩 QA Chain
prompt_template = """
You are an intelligent assistant that answers questions based on the provided context.

Context:
{context}

Question:
{question}

Answer:
"""
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever, chain_type_kwargs={"prompt": PROMPT})

# 🎯 Query input
query = st.text_input("🎯 Ask your question:")

if st.button("Run Query") and query:
    st.info("Running your query... please wait ⏳")
    try:
        response = qa_chain.invoke({"query": query})
        st.success(response["result"])
    except Exception as e:
        st.error(f"⚠️ Error: {e}")

st.markdown("---")
st.caption("Built with ❤️ by Aditi | Powered by OpenAI • FAISS • Streamlit")

