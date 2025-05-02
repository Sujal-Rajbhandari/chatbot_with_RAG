# from langchain_community.document_loaders import UnstructuredPDFLoader
# from langchain_ollama import OllamaEmbeddings 
# from langchain_text_splitters import RecursiveCharacterTextSplitter
# from langchain_community.vectorstores import Chroma
# from langchain.prompts import ChatPrompTemplate, PromptTemplate
# from langchain_core.output_parsers import StrOutputParser
# from langchain_ollama.chat_models import ChatOllama 
# from langchain_core.runnables import RunnablePassthrough
# from langchain.retruevers.multi_query import MultiQueryRetriever

# import os 



import os
import pytesseract
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings
from langchain_ollama.llms import OllamaLLM
from langchain.chains import RetrievalQA

pdfs_directory = 'pdfs/'

# Embedding & LLM
embeddings = OllamaEmbeddings(model="deepseek-r1:1.5b")
model = OllamaLLM(model="deepseek-r1:1.5b")

# Upload PDF
def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())

# OCR fallback
def extract_text_from_images(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

# Vector Store Creation
def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    # If the loader doesn't extract meaningful text, use OCR
    combined_text = " ".join([doc.page_content.strip() for doc in documents])
    if not combined_text or len(combined_text) < 30:
        print("No meaningful text found, trying OCR...")
        text = extract_text_from_images(file_path)
        documents = [Document(page_content=text, metadata={})]

    # Split text into manageable chunks
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=300,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)

    # Embed and store vectors
    db = FAISS.from_documents(chunked_docs, embeddings)
    return db

# Create RAG Pipeline
def create_rag_pipeline(db):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="map_reduce",  # handles long contexts better
        return_source_documents=True
    )
    return rag_chain

# Ask a question
def question_pdf(question, rag_chain):
    result = rag_chain.invoke({"query": question})
    return result['result']
