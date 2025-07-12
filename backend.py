import time
import streamlit as st
import requests as requests
from fastapi import FastAPI, File, UploadFile, Form, HTTPException, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from datetime import datetime
from dotenv import load_dotenv
from uuid import uuid4
import base64
import os

from config import GOOGLE_API_KEY
from history import save_history, get_history, clear_history
from pdf_utils import extract_text_from_pdfs
from vectorstore_utils import chunk_text, create_vector_store, load_vector_store
from llm_utils import get_chain 
from performance_monitor import PerformanceMonitor
performance_monitor = PerformanceMonitor()


app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/upload_pdfs/")
@performance_monitor.timing_decorator("upload_pdfs_endpoint")
async def upload_pdfs(files: list[UploadFile] = File(...), session_id: str = Form(None)):
    session_id = session_id or str(uuid4())
    
    # Time PDF processing
    start_time = time.time()
    ocr_text, text, pdf_names = await extract_text_from_pdfs(files)
    pdf_processing_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["pdf_processing"].append(pdf_processing_time)
    print(f"[PERFORMANCE] PDF Processing: {pdf_processing_time:.2f}ms")
    
    if not text:
        return JSONResponse(status_code=400, content={"error": "No text extracted from PDFs."})
    
    # Time text chunking
    start_time = time.time()
    chunks = chunk_text(text)
    chunking_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["text_chunking"].append(chunking_time)
    print(f"[PERFORMANCE] Text Chunking: {chunking_time:.2f}ms")
    
    if not chunks:
        return JSONResponse(status_code=400, content={"error": "Failed to split text into chunks."})
    
    # Time vector store creation
    start_time = time.time()
    create_vector_store(chunks, session_id)
    vector_store_time = (time.time() - start_time) * 1000
    performance_monitor.metrics["vector_store_creation"].append(vector_store_time)
    print(f"[PERFORMANCE] Vector Store Creation: {vector_store_time:.2f}ms")
    
    return {
        "session_id": session_id, 
        "pdf_names": pdf_names, 
        "chunks": len(chunks),
        'OCR': ocr_text,
        'text': text,
        "performance_metrics": {
            "pdf_processing_ms": pdf_processing_time,
            "text_chunking_ms": chunking_time,
            "vector_store_creation_ms": vector_store_time
        }
    }

@app.post("/upload_pdfs_base64/")
async def upload_pdfs_base64(
    files_base64: list[str] = Body(...),
    filenames: list[str] = Body(...),
    session_id: str = Form(None)
):
    session_id = session_id or str(uuid4())
    temp_files = []
    try:
        # Decode and save each base64 PDF
        for b64, fname in zip(files_base64, filenames):
            pdf_bytes = base64.b64decode(b64)
            temp_path = f"pdf_output/{session_id or 'default'}/{fname}"
            os.makedirs(os.path.dirname(temp_path), exist_ok=True)
            with open(temp_path, "wb") as f:
                f.write(pdf_bytes)
            # Create a file-like object for processing
            class DummyUploadFile:
                def __init__(self, path, filename):
                    self.file = open(path, "rb")
                    self.filename = filename
                async def read(self):
                    self.file.seek(0)
                    return self.file.read()
            temp_files.append(DummyUploadFile(temp_path, fname))
        # --- Performance metrics ---
        # Time PDF processing
        start_time = time.time()
        ocr_text, text, pdf_names = await extract_text_from_pdfs(temp_files, session_id=session_id)
        pdf_processing_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["pdf_processing"].append(pdf_processing_time)
        print(f"[PERFORMANCE] PDF Processing: {pdf_processing_time:.2f}ms")

        # Clean up file handles
        for f in temp_files:
            f.file.close()

        if not text:
            return JSONResponse(status_code=400, content={"error": "No text extracted from PDFs."})

        # Time text chunking
        start_time = time.time()
        chunks = chunk_text(text)
        chunking_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["text_chunking"].append(chunking_time)
        print(f"[PERFORMANCE] Text Chunking: {chunking_time:.2f}ms")

        if not chunks:
            return JSONResponse(status_code=400, content={"error": "Failed to split text into chunks."})

        # Time vector store creation
        start_time = time.time()
        create_vector_store(chunks, session_id)
        vector_store_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["vector_store_creation"].append(vector_store_time)
        print(f"[PERFORMANCE] Vector Store Creation: {vector_store_time:.2f}ms")

        return {
            "session_id": session_id,
            "pdf_names": pdf_names,
            "chunks": len(chunks),
            'OCR': ocr_text,
            'text': text,
            "performance_metrics": {
                "pdf_processing_ms": pdf_processing_time,
                "text_chunking_ms": chunking_time,
                "vector_store_creation_ms": vector_store_time
            }
        }
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/chat/")
@performance_monitor.timing_decorator("chat_endpoint")
async def chat(query: str = Form(...), session_id: str = Form(...)):
    try:
        # Time vector store loading
        start_time = time.time()
        vector_store = load_vector_store(session_id)
        vector_load_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["vector_store_loading"].append(vector_load_time)
        print(f"[PERFORMANCE] Vector Store Loading: {vector_load_time:.2f}ms")
        
        # Time similarity search
        start_time = time.time()
        docs = vector_store.similarity_search(query,k=2)
        search_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["similarity_search"].append(search_time)
        print(f"[PERFORMANCE] Similarity Search: {search_time:.2f}ms")
        
        # Time LLM inference
        start_time = time.time()
        chain = get_chain()
        response = chain({"input_documents": docs, "question": query}, return_only_outputs=True)
        llm_time = (time.time() - start_time) * 1000
        performance_monitor.metrics["llm_inference"].append(llm_time)
        print(f"[PERFORMANCE] LLM Inference: {llm_time:.2f}ms")
        
        answer = response['output_text']
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        pdf_names = "N/A"
        save_history(session_id, query, answer, "Google AI", timestamp, pdf_names)
        
        return {
            "answer": answer, 
            "timestamp": timestamp, 
            "session_id": session_id,
            "performance_metrics": {
                "vector_store_loading_ms": vector_load_time,
                "similarity_search_ms": search_time,
                "llm_inference_ms": llm_time
            }
        }
    except Exception as e:
        return JSONResponse(status_code=404, content={"error": f"No vector store found for session: {str(e)}"})

@app.post("/reset/")
async def reset(session_id: str = Form(...)):
    clear_history(session_id)
    # Optionally, remove FAISS index for session
    import shutil
    try:
        shutil.rmtree(f"faiss_index/{session_id}")
    except Exception:
        pass
    return {"status": "reset", "session_id": session_id}

@app.get("/history/")
async def history(session_id: str):
    history = get_history(session_id)
    return {"session_id": session_id, "history": history}

@app.get("/performance_metrics/")
async def get_performance_metrics():
    """Get aggregated performance metrics"""
    return performance_monitor.get_metrics_summary()

@app.get("/performance_metrics/save/")
async def save_performance_metrics():
    """Save current metrics to file"""
    performance_monitor.save_metrics_to_file()
    return {"message": "Metrics saved successfully"}

@app.get("/")
def root():
    return {"status": "FastAPI backend running"}


