import streamlit as st
import main as main

st.title("Chat with PDF using Deepseek")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

if uploaded_file:
    main.upload_pdf(uploaded_file)
    pdf_path = main.pdfs_directory + uploaded_file.name
    db = main.create_vector_store(pdf_path)
    rag_chain = main.create_rag_pipeline(db)

    st.success("PDF processed successfully. Ask your questions below.")

    question = st.chat_input("Ask something about your PDF")
    if question:
        st.chat_message("user").write(question)
        answer = main.question_pdf(question, rag_chain)
        st.chat_message("assistant").write(answer)
