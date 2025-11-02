import streamlit as st
from openai import OpenAI


#  Load API Key from Streamlit Secrets

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("🚨 Missing OpenAI API key. Please add it under 'Secrets' in Streamlit Cloud.")
    st.stop()

client = OpenAI(api_key=OPENAI_API_KEY)


# Streamlit Page Config

st.set_page_config(
    page_title="GenQuery – AI Assistant",
    page_icon="🎬",
    layout="centered"
)


# App Header

st.markdown(
    """
    <h1 style='text-align: center; color: #F5C518;'>🎬 GenQuery – AI Assistant</h1>
    <p style='text-align: center; color: #bbb;'>Built with <b>OpenAI GPT</b> + <b>Streamlit</b></p>
    """,
    unsafe_allow_html=True
)

st.divider()

st.subheader("🎥 Try queries like:")
st.markdown("""
- *Top 5 sci-fi movies after 2015*  
- *Who directed the most movies in 2020?*  
- *Highest-rated action movies by genre*
""")


# User Query

query = st.text_input("🔍 Enter your movie-related question:")

if query:
    st.info("Running your query... please wait ⏳")
    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an expert movie data assistant that helps users explore films, genres, and directors."},
                {"role": "user", "content": query}
            ],
            temperature=0.5
        )

        st.success(response.choices[0].message.content.strip())
    except Exception as e:
        st.error(f"⚠️ Error while processing your query: {e}")
else:
    st.warning("Please enter a question to begin 🎯")
