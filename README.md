## GenQuery – LLM-Based SQL Interpreter

GenQuery is an intelligent LLM-powered SQL assistant that lets you query databases in plain English.
It uses Retrieval-Augmented Generation (RAG), OpenAI and LLaMA-3 models via LangChain, along with FAISS and Hugging Face embeddings, to generate accurate, context-aware SQL queries and execute them instantly.

#Features

- Natural-Language to SQL Conversion – Translate plain English into executable SQL queries.
- RAG-Powered Context Understanding – Uses FAISS vector retrieval for semantic context and better reasoning.
- LLM Flexibility – Works with both OpenAI GPT models and Meta LLaMA-3 for SQL generation.
- Interactive Streamlit UI – Intuitive and responsive dashboard for querying and visualization.
- Extensible Design – Plug in your own datasets or LLMs with minimal changes.


# Installation

Clone repository
git clone https://github.com/aditirpatil11/GenQuery.git
cd GenQuery

# Install dependencies
pip install -r requirements.txt
Run locally
streamlit run App/app.py

# How It Works

- Embedding Generation – The IMDB schema and sample data are embedded using Hugging Face models.
- RAG Pipeline – FAISS retrieves relevant schema chunks based on your natural-language query.
- LLM Reasoning – OpenAI / LLaMA-3 interprets intent and generates valid SQL queries.
- Execution & Visualization – The app executes SQL and displays results in an interactive Streamlit UI.



# Deployment
Streamlit Cloud → https://streamlit.io/cloud
Hugging Face Spaces → Gradio / Streamlit runtime

# Tech Stack
- LLMs: OpenAI GPT, Meta LLaMA-3
- Retrieval	RAG (LangChain + FAISS)
- Embeddings	Hugging Face Sentence Transformers
- Backend	Python
- Frontend	Streamlit
- Data	IMDB dataset (TSV → SQLite)
