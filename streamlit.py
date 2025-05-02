import streamlit as st
import rag as main  

st.set_page_config(page_title="Chat with PDFs", layout="centered")
st.title(" Chat with PDFs using Deepseek + RAG")

# File upload
uploaded_file = st.file_uploader(
    "Upload a PDF document",
    type="pdf",
    accept_multiple_files=False
)

# Step-by-step checks
if uploaded_file:
    st.success(f"Uploaded file: {uploaded_file.name}")

    try:
        # Save PDF
        main.upload_pdf(uploaded_file)
        pdf_path = main.pdfs_directory + uploaded_file.name

        # Vector store and RAG
        with st.spinner("Processing PDF and creating vector store..."):
            db = main.create_vector_store(pdf_path)
            rag_chain = main.create_rag_pipeline(db)
            st.success("PDF processed and ready for Q&A!")

        # Chat
        question = st.chat_input("Ask something about your PDF")

        if question:
            st.chat_message("user").write(question)

            with st.spinner("Thinking..."):
                answer = main.question_pdf(question, rag_chain)
                st.chat_message("assistant").write(answer)

    except Exception as e:
        st.error("An error occurred:")
        st.exception(e)
else:
    st.info("Please upload a PDF to begin.")
