import streamlit as st
import langchain_helper as lch
import textwrap

st.title("Youtube Assistant")

with st.sidebar:
  with st.form(key="my_form"):
    youtube_url = st.sidebar.text_area(
      label="What is the Youtube video URL?",
      max_chars=100
    )
    query = st.sidebar.text_area(
      label="Ask me about the video?",
      max_chars=50,
      key="query"
    )
    submitted = st.form_submit_button("Submit")
    


if query and youtube_url and submitted:
  db = lch.create_vector_db_from_youtube(youtube_url)
  response, docs = lch.get_response_from_query(db, query)
  st.subheader("Answer:")
  st.text(
    textwrap.fill(response, width=80)
  )