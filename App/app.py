from flask import Flask, render_template, request, jsonify
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
import os
import pickle

# Initialize Flask app
app = Flask(__name__)

# Load FAISS index if exists
if os.path.exists("faiss_index.pkl"):
    with open("faiss_index.pkl", "rb") as f:
        vectorstore = pickle.load(f)
else:
    embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
    docs = ["This is a placeholder database. Add your data using rag_imdb folder."]
    vectorstore = FAISS.from_texts(docs, embeddings)
    with open("faiss_index.pkl", "wb") as f:
        pickle.dump(vectorstore, f)

# Initialize model
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)
qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=vectorstore.as_retriever())

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/query", methods=["POST"])
def query():
    user_input = request.form["query"]
    response = qa_chain.run(user_input)
    return jsonify({"response": response})

if __name__ == "__main__":
    app.run(debug=True)
