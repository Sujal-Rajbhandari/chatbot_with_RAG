# Chat with PDF using Gemini and RAG

This project is a Streamlit-based PDF question-answering application. It allows users to upload a PDF document and ask questions about its content. The system uses **Gemini 1.5 Flash**, **Google Generative AI Embeddings**, **LangChain**, **FAISS**, and **RAG** to retrieve relevant document chunks and generate answers.

The application also supports scanned/image-based PDFs using OCR with **Tesseract** and **pdf2image**.

---

## Features

* Upload PDF documents through a Streamlit interface
* Extract text from normal text-based PDFs
* Use OCR for scanned or image-based PDFs
* Split PDF content into smaller text chunks
* Generate embeddings using Google Generative AI Embeddings
* Store document chunks in a FAISS vector database
* Retrieve relevant chunks using similarity search
* Ask questions about the uploaded PDF
* Generate answers using Gemini 1.5 Flash
* Supports OpenAI models as an optional alternative

---

## Project Structure

```bash
chat-with-pdf/
│
├── main.py
├── streamlit.py
├── .env
├── requirements.txt
├── pdfs/
└── README.md
```

---

## Technologies Used

* Python
* Streamlit
* LangChain
* Gemini 1.5 Flash
* Google Generative AI Embeddings
* FAISS
* PyPDFLoader
* Tesseract OCR
* pdf2image
* python-dotenv
* OpenAI API support optional

---

## How It Works

### 1. Upload PDF

The user uploads a PDF file using the Streamlit interface.

### 2. Save PDF

The uploaded PDF is saved inside the `pdfs/` directory.

### 3. Text Extraction

The system first tries to load and extract text using `PyPDFLoader`.

If the extracted text is empty or too short, the system assumes the PDF may be scanned or image-based and applies OCR using `pytesseract`.

### 4. Text Chunking

The extracted text is split into smaller chunks using `RecursiveCharacterTextSplitter`.

Chunk settings:

```python
chunk_size = 1000
chunk_overlap = 200
```

### 5. Vector Store Creation

Each chunk is converted into embeddings using Google Generative AI Embeddings and stored in a FAISS vector database.

### 6. RAG Pipeline

A retrieval-based QA chain is created using LangChain’s `RetrievalQA`.

The retriever fetches the top 5 most relevant chunks for the user’s question.

### 7. Answer Generation

Gemini 1.5 Flash generates an answer based on the retrieved document context.

---

## Installation

### 1. Clone the repository

```bash
git clone <your-repository-url>
cd chat-with-pdf
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

Activate the virtual environment.

For Windows:

```bash
venv\Scripts\activate
```

For macOS/Linux:

```bash
source venv/bin/activate
```

---

## Install Dependencies

Create a `requirements.txt` file:

```txt
streamlit
python-dotenv
pytesseract
pdf2image
langchain
langchain-community
langchain-text-splitters
langchain-google-genai
langchain-openai
faiss-cpu
pypdf
```

Then install the dependencies:

```bash
pip install -r requirements.txt
```

---

## Environment Variables

Create a `.env` file in the project root directory:

```env
GOOGLE_API_KEY=your_google_api_key_here
```

If you want to use OpenAI instead of Gemini, you can also add:

```env
OPENAI_API_KEY=your_openai_api_key_here
```

---

## OCR Setup

This project uses OCR for scanned PDFs, so you need to install **Tesseract OCR** and **Poppler**.

---

### Install Tesseract OCR

For macOS:

```bash
brew install tesseract
```

For Linux:

```bash
sudo apt install tesseract-ocr
```

For Windows:

Download and install Tesseract OCR, then add it to your system PATH.

---

### Install Poppler

`pdf2image` requires Poppler to convert PDF pages into images.

For macOS:

```bash
brew install poppler
```

For Linux:

```bash
sudo apt install poppler-utils
```

For Windows:

Download Poppler and add the `bin` folder to your system PATH.

---

## How to Run

Run the Streamlit app:

```bash
streamlit run streamlit.py
```

Then open the local URL shown in the terminal.

Usually:

```bash
http://localhost:8501
```

---

## Example Usage

1. Upload a PDF document.
2. Wait for the document to be processed.
3. Enter a question such as:

```text
What is this document about?
```

```text
Summarize the key points of the PDF.
```

```text
What are the important dates mentioned in the document?
```

4. The app will return an answer based on the PDF content.

---

## Main Components

### `upload_pdf(file)`

Saves the uploaded PDF file into the `pdfs/` directory.

### `extract_text_from_images(pdf_path)`

Converts scanned PDF pages into images and extracts text using Tesseract OCR.

### `create_vector_store(file_path)`

Loads the PDF, extracts text, applies OCR if needed, splits the document into chunks, creates embeddings, and stores them in FAISS.

### `create_rag_pipeline(db)`

Creates a RAG question-answering pipeline using a FAISS retriever and Gemini 1.5 Flash.

### `question_pdf(question, rag_chain)`

Takes the user’s question and returns an answer generated from the retrieved PDF content.

---

## Example Code Flow

```text
Upload PDF
    ↓
Save PDF to local folder
    ↓
Extract text using PyPDFLoader
    ↓
If text is empty, apply OCR
    ↓
Split text into chunks
    ↓
Create embeddings
    ↓
Store chunks in FAISS
    ↓
Retrieve relevant chunks
    ↓
Generate answer using Gemini
```

---

## Limitations

* Accuracy depends on the quality of the PDF.
* Scanned PDFs may produce lower-quality answers if OCR quality is poor.
* Large PDFs may take more time to process.
* The app currently rebuilds the vector store every time a PDF is uploaded.
* Answers depend on the retrieved chunks, so poor retrieval can affect response quality.
* The app does not currently show source pages or document references.

---

## Future Improvements

* Add chat history
* Add source page references
* Add support for multiple PDFs
* Cache vector stores to avoid reprocessing the same PDF
* Add downloadable answers
* Add better OCR language support
* Add document summary generation
* Add support for DOCX and TXT files
* Deploy the app on AWS or Streamlit Cloud
* Add user authentication
* Add better error handling and logging

---

## Author

**Sujal Rajbhandari**
AI/ML Engineer
Kathmandu, Nepal
