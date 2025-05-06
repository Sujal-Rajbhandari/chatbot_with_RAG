import os
from dotenv import load_dotenv
import pytesseract
from pdf2image import convert_from_path
from langchain.schema import Document
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings


load_dotenv()

pdfs_directory = 'pdfs/'

embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
model = ChatGoogleGenerativeAI(model="models/gemini-1.5-flash", temperature=0)
# embeddings = OpenAIEmbeddings() 
# model = ChatOpenAI(model_name="gpt-4", temperature=0)

def upload_pdf(file):
    os.makedirs(pdfs_directory, exist_ok=True)
    with open(os.path.join(pdfs_directory, file.name), "wb") as f:
        f.write(file.getbuffer())

def extract_text_from_images(pdf_path):
    images = convert_from_path(pdf_path)
    text = ""
    for image in images:
        text += pytesseract.image_to_string(image)
    return text

def create_vector_store(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    combined_text = " ".join([doc.page_content.strip() for doc in documents])
    if not combined_text or len(combined_text) < 30:
        print("No meaningful text found, trying OCR...")
        text = extract_text_from_images(file_path)
        documents = [Document(page_content=text, metadata={})]

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        add_start_index=True
    )
    chunked_docs = text_splitter.split_documents(documents)

    db = FAISS.from_documents(chunked_docs, embeddings)
    return db

def create_rag_pipeline(db):
    retriever = db.as_retriever(search_kwargs={"k": 5})
    rag_chain = RetrievalQA.from_chain_type(
        llm=model,
        retriever=retriever,
        chain_type="map_reduce",  
        return_source_documents=True
    )
    return rag_chain

def question_pdf(question, rag_chain):
    result = rag_chain.invoke({"query": question})
    return result['result']

