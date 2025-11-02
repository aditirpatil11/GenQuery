import streamlit as st
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

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

# LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3)

# Prompt template
prompt = ChatPromptTemplate.from_template("""
You are a movie data analyst with access to IMDb-like data.
Use the provided retrieved context to answer the user's question.

Context:
{context}

Question:
{question}

Answer clearly and concisely.
""")

def rag_answer(question):
    """Manual RAG retrieval + generation"""
    try:
        docs = retriever.get_relevant_documents(question)
        context = "\n\n".join([d.page_content for d in docs])
        full_prompt = prompt.format(context=context, question=question)
        response = llm.invoke(full_prompt)
        return response.content
    except Exception as e:
        return f"Error during RAG processing: {e}"

if query:
    with st.spinner("🔍 Thinking... generating intelligent response..."):
        answer = rag_answer(query)
        st.success("✅ Answer:")
        st.write(answer)

st.markdown("---")
st.caption("Built using Streamlit • LangChain • OpenAI • HuggingFace • FAISS")

