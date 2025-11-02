import streamlit as st
import os
from openai import OpenAI


# Load API Key from Streamlit Secrets

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    st.error("🚨 Missing OpenAI API key. Please add it under 'Secrets' in Streamlit Cloud.")
    st.stop()

os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
client = OpenAI(api_key=OPENAI_API_KEY)


# Streamlit Page Config

st.set_page_config(
    page_title="GenQuery – AI Movie Data Assistant",
    page_icon="🎬",
    layout="centered",
)


# Header + Description

st.markdown(
    """
    <h1 style='text-align: center; color: #F5C518;'>🎬 GenQuery – AI Assistant</h1>
    <p style='text-align: center; color: #bbbbbb;'>Built with <b>LangChain</b> • <b>OpenAI</b> • <b>FAISS</b> • <b>Streamlit</b></p>
    """,
    unsafe_allow_html=True
)

st.markdown("---")
st.subheader("🎥 Try queries like:")
st.markdown("""
- *Top 5 sci-fi movies after 2015*  
- *Who directed the most movies in 2020?*  
- *Average IMDb rating by genre*
""")


# Query Input

query = st.text_input("🔍 Enter your question:")

if query:
    st.info("Running your query... please wait ⏳")

    try:
        # Send the query to OpenAI API
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a movie data assistant that answers questions based on IMDB-style data."},
                {"role": "user", "content": query}
            ],
            temperature=0.3
        )

        answer = response.choices[0].message.content.strip()
        st.success(f"🎞️ **Answer:**\n\n{answer}")

    except Exception as e:
        st.error(f" Error while processing your request: {e}")

else:
    st.warning("Please enter a question to begin 🎯")
