import os
import shutil
from typing import List

import fitz
import faiss
import numpy as np
import requests

from dotenv import load_dotenv
from fastapi import FastAPI, File, UploadFile, Request, Form
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from sentence_transformers import SentenceTransformer


load_dotenv()

app = FastAPI(title="Simple RAG Document Search Assistant")
templates = Jinja2Templates(directory="templates")

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")

documents = []
chunks = []
chunk_sources = []
chat_history = []
index = None


def extract_text_from_pdf(file_path: str) -> str:
    text = ""

    with fitz.open(file_path) as pdf:
        for page_num, page in enumerate(pdf, start=1):
            page_text = page.get_text()
            text += f"\n\n[Page {page_num}]\n{page_text}"

    return text


def split_text(text: str, chunk_size: int = 900, overlap: int = 150) -> List[str]:
    result = []
    start = 0

    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]

        if chunk.strip():
            result.append(chunk.strip())

        start += chunk_size - overlap

    return result


def build_vector_index():
    global index

    if not chunks:
        index = None
        return

    embeddings = embedding_model.encode(chunks)
    embeddings = np.array(embeddings).astype("float32")

    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)


def retrieve_chunks(question: str, top_k: int = 4):
    if index is None or not chunks:
        return []

    query_embedding = embedding_model.encode([question])
    query_embedding = np.array(query_embedding).astype("float32")

    distances, indices = index.search(query_embedding, top_k)

    results = []

    for distance, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue

        distance_value = float(distance)
        similarity = 1 / (1 + distance_value)
        similarity_percentage = round(similarity * 100, 2)

        results.append({
            "text": chunks[idx],
            "source": chunk_sources[idx],
            "score": distance_value,
            "similarity": similarity_percentage
        })

    return results


def generate_answer(question: str, retrieved, history):
    if not GEMINI_API_KEY:
        return "Gemini API key missing. Please add GEMINI_API_KEY to your .env file."

    if not retrieved:
        return "I could not find this in the uploaded documents."

    context = "\n\n".join(
        [
            f"Source: {item['source']}\nSimilarity: {item['similarity']}%\nContent:\n{item['text']}"
            for item in retrieved
        ]
    )

    conversation_context = "\n\n".join(
        [
            f"User: {item['question']}\nAssistant: {item['answer']}"
            for item in history[-5:]
        ]
    )

    prompt = f"""
You are a document question-answering assistant.

Answer the user's current question using ONLY the uploaded document context.

You may use the recent conversation only to understand follow-up questions.
However, factual answers must still come only from the uploaded document context.

Strict rules:
1. Use only the uploaded document context for facts.
2. Do not use outside knowledge.
3. If the answer is not clearly present in the context, say exactly:
   "I could not find this in the uploaded documents."
4. Keep the answer clear and concise.
5. Mention the source file/page when possible.

Recent Conversation:
{conversation_context}

Uploaded Document Context:
{context}

Current Question:
{question}

Answer:
"""

    url = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent"

    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": GEMINI_API_KEY
    }

    payload = {
        "contents": [
            {
                "parts": [
                    {
                        "text": prompt
                    }
                ]
            }
        ]
    }

    try:
        response = requests.post(
            url,
            headers=headers,
            json=payload,
            timeout=30
        )

        response.raise_for_status()
        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as e:
        return f"Gemini API error: {str(e)}"


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    global chat_history

    # Chat history resets when page is opened/reloaded.
    chat_history = []

    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents,
        "answer": None,
        "retrieved": [],
        "question": "",
        "chat_history": chat_history
    })


@app.post("/upload", response_class=HTMLResponse)
async def upload_documents(request: Request, files: List[UploadFile] = File(...)):
    global documents, chunks, chunk_sources, chat_history

    documents = []
    chunks = []
    chunk_sources = []
    chat_history = []

    skipped_files = []

    for file in files:
        if not file.filename.lower().endswith(".pdf"):
            skipped_files.append(file.filename)
            continue

        file_path = os.path.join(UPLOAD_DIR, file.filename)

        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        text = extract_text_from_pdf(file_path)

        if not text.strip():
            skipped_files.append(file.filename)
            continue

        file_chunks = split_text(text)

        documents.append(file.filename)

        for chunk in file_chunks:
            chunks.append(chunk)

            page_label = "Unknown page"

            if "[Page " in chunk:
                try:
                    page_label = "Page " + chunk.split("[Page ")[1].split("]")[0]
                except Exception:
                    page_label = "Unknown page"

            chunk_sources.append(f"{file.filename}, {page_label}")

    build_vector_index()

    message = f"Uploaded {len(documents)} document(s). Created {len(chunks)} chunks."

    if skipped_files:
        message += f" Skipped {len(skipped_files)} file(s): {', '.join(skipped_files)}."

    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents,
        "answer": message,
        "retrieved": [],
        "question": "",
        "chat_history": chat_history
    })


@app.post("/ask", response_class=HTMLResponse)
async def ask_question(request: Request, question: str = Form(...)):
    retrieved = retrieve_chunks(question)
    answer = generate_answer(question, retrieved, chat_history)

    chat_history.append({
        "question": question,
        "answer": answer
    })

    return templates.TemplateResponse("index.html", {
        "request": request,
        "documents": documents,
        "answer": answer,
        "retrieved": retrieved,
        "question": question,
        "chat_history": chat_history
    })