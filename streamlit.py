import streamlit as st
from main import upload_pdf, create_vector_store, create_rag_pipeline, question_pdf
import os

st.set_page_config(page_title="Chat with PDF", layout="wide")
st.title("Chat with your PDF using Gemini ")

uploaded_file = st.file_uploader("Upload a PDF file", type="pdf")

if uploaded_file:
    upload_pdf(uploaded_file)
    file_path = os.path.join("pdfs", uploaded_file.name)

    db = create_vector_store(file_path)
    rag_chain = create_rag_pipeline(db)

    st.success("Ask a question below!")

    user_question = st.text_input("Enter your question:")

    if user_question:
        response = question_pdf(user_question, rag_chain)
        st.markdown("### Answer:")
        st.write(response)
