from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse, JSONResponse
from typing import List
from pydantic import BaseModel

from rag_system import RAGSystem

app = FastAPI(title="RAG System")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class QueryRequest(BaseModel):
    question: str
    k: int = 3
    stream_delay: float = 0.1

class AddDocumentsRequest(BaseModel):
    files: List[str]

# Initialize RAG system
# Optional: pass recreate_index=True to rebuild vector DB from PDFs at startup
rag_system = RAGSystem(recreate_index=False)

@app.get("/")
async def root():
    return {"message": "RAG System API"}

@app.post("/query-stream")
async def query_rag_system_stream(request: QueryRequest):
    """Streaming query endpoint"""
    try:
        # StreamingResponse expects an async generator
        return StreamingResponse(
            rag_system.stream_query(
                question=request.question,
                k=request.k,
                stream_delay=request.stream_delay
            ),
            media_type="text/event-stream",
            headers={
                'Cache-Control': 'no-cache',
                'Connection': 'keep-alive',
                'X-Accel-Buffering': 'no',
                'Content-Type': 'text/event-stream; charset=utf-8',
                'Transfer-Encoding': 'chunked',  # Explicit chunked encoding
                'X-Content-Type-Options': 'nosniff',  # Prevent MIME sniffing
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/add-documents")
async def add_documents(request: AddDocumentsRequest):
    """Add new documents to PostgreSQL vector store"""
    try:
        rag_system.add_documents(request.files)
        return {"message": "Documents added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    """Catch-all exception handler"""
    return JSONResponse(
        status_code=500,
        content={"message": f"An error occurred: {str(exc)}"}
    )