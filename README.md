# GenQuery – LLM-Based SQL Interpreter

GenQuery is an intelligent LLM-powered SQL assistant that lets you query databases in plain English.
It uses Retrieval-Augmented Generation (RAG), OpenAI and LLaMA-3 models via LangChain, along with FAISS and Hugging Face embeddings, to generate accurate, context-aware SQL queries and execute them instantly.

# Features

- Natural-Language to SQL Conversion – Translate plain English into executable SQL queries.
- RAG-Powered Context Understanding – Uses FAISS vector retrieval for semantic context and better reasoning.
- LLM Flexibility – Works with both OpenAI GPT models and Meta LLaMA-3 for SQL generation.
- Interactive Flask Web UI – Lightweight local web interface with clean design (no external cloud dependencies).
- Extensible Design – Replace the IMDB dataset with your own text or database easily.


# Installation
- Clone the repository
git clone https://github.com/aditirpatil11/GenQuery.git
cd GenQuery

- Install dependencies
pip install -r requirements.txt
Create a .env file in the root directory and add your API key:
OPENAI_API_KEY=sk-xxxxxxxxxxxxxxxx

# Deployment
Local Flask Deployment
- GenQuery was deployed locally using Flask, providing a responsive web interface hosted at
- http://127.0.0.1:5000

Steps followed:
- Installed dependencies using pip install -r requirements.txt
- Created a .env file with the OpenAI API key
- Started the local server via python3 App/app.py
- Accessed the Flask interface in a browser for querying and testing

# How It Works

- Data Embedding – The IMDB movie data (imdb.txt) is converted into vector embeddings using Hugging Face models.
- RAG Pipeline – FAISS retrieves the most relevant text chunks based on the query.
- LLM Reasoning – OpenAI GPT interprets intent, ranks or summarizes results, and produces natural-language answers.
- Flask Interface – Displays the query and AI response dynamically in your local web app.

# Dataset
- Since the official IMDb dataset (imdb.db) is extremely large, a lightweight subset (imdb.txt) was used for this demo to ensure smooth embedding and retrieval performance.
- The original full IMDb dataset can be downloaded from:
🔗 IMDb Datasets – Official Website
- You can integrate the real dataset by converting the .tsv files into an SQLite database and adjusting the embedding pipeline accordingly.



# Tech Stack
- LLMs: OpenAI GPT, Meta LLaMA-3
- Retrieval	RAG (LangChain + FAISS)
- Embeddings	Hugging Face Sentence Transformers
- Backend	Python and Flask 
- Frontend	HTML/CSS 
- Data	IMDB dataset 
