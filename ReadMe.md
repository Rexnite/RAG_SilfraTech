# Document Search Assistant (RAG)

## Overview

This project is a Retrieval-Augmented Generation (RAG) based Document Search Assistant that enables users to upload PDF documents and ask natural language questions about their contents.

The system extracts text from uploaded PDFs, splits the text into chunks, generates semantic embeddings using Sentence Transformers, stores the embeddings in a FAISS vector index, retrieves the most relevant chunks for a user query, and uses Gemini 2.5 Flash to generate grounded responses.

The application is built using FastAPI and provides a web interface for document upload, querying, and answer visualization.

---

## Features

### Core Features

* Upload one or more PDF documents
* Automatic text extraction using PyMuPDF
* Chunking with overlap for improved retrieval quality
* Semantic embedding generation using Sentence Transformers
* FAISS vector database for similarity search
* Retrieval-Augmented Generation (RAG) pipeline
* Gemini 2.5 Flash integration for answer generation
* Source attribution for retrieved content

### Bonus Features

* Multiple PDF support
* Similarity score display
* Conversational chat history
* Context-aware follow-up questions
* Source citations
* Docker support

---

## System Architecture

```text
PDF Upload
    ↓
Text Extraction (PyMuPDF)
    ↓
Chunking
    ↓
Sentence Transformer Embeddings
    ↓
FAISS Vector Index
    ↓
Top-K Semantic Retrieval
    ↓
Gemini 2.5 Flash
    ↓
Answer Generation
```

---

## Technology Stack

### Backend

* Python
* FastAPI
* Uvicorn

### Document Processing

* PyMuPDF (fitz)

### Embeddings

* Sentence Transformers
* all-MiniLM-L6-v2

### Vector Database

* FAISS

### Large Language Model

* Gemini 2.5 Flash API

### Frontend

* HTML
* CSS
* Jinja2 Templates

---

## Project Structure

```text
RAG/
│
├── main.py
├── requirements.txt
├── README.md
├── Dockerfile
├── .dockerignore
│
├── templates/
│   └── index.html
│
├── uploads/
│
└── sample_documents/
```

---

## Installation

### 1. Clone the repository

```bash
git clone <repository-url>
cd RAG
```

### 2. Create a virtual environment

```bash
python -m venv venv
```

### 3. Activate the virtual environment

Windows:

```bash
venv\Scripts\activate
```

Linux/Mac:

```bash
source venv/bin/activate
```

### 4. Install dependencies

```bash
pip install -r requirements.txt 
```
Must
### 5. Configure API Key

Create a `.env` file:

```env
GEMINI_API_KEY=YOUR_API_KEY
```

### 6. Run the application

```bash
python -m uvicorn main:app --port 8001
```

Open:

```text
http://127.0.0.1:8001
```

---

## Docker Usage

Build the image:

```bash
docker build -t rag-assistant .
```

Run the container:

```bash
docker run -p 8000:8000 rag-assistant
```

Open:

```text
http://localhost:8000
```

---

## Example Workflow

1. Upload one or more PDF documents.
2. The application extracts text and generates embeddings.
3. Ask a question related to the uploaded documents.
4. Relevant chunks are retrieved from FAISS.
5. Gemini generates an answer using the retrieved context.
6. Sources and similarity scores are displayed.

---

## Sample Questions

* What are Python dictionaries?
* Explain list comprehensions.
* What is inheritance in object-oriented programming?
* Summarize the uploaded document.
* What are the key concepts discussed in Chapter 3?

---

## Limitations

* Supports PDF documents only.
* Uploaded documents are stored locally.
* Chat history is session-based and resets when the page is refreshed.
* Answers are limited to the information available in uploaded documents.
*run pip install -r requirements.txt to run Run.bat
---

## Future Improvements

* Persistent vector storage
* User authentication
* Database integration
* Streaming responses
* PDF page previews
* Advanced document management
* Multi-user support

---

## Author

Pranjal Sarkar

Internship Assignment Submission

Built using FastAPI, FAISS, Sentence Transformers, and Gemini 2.5 Flash.
