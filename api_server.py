#!/usr/bin/env python3
"""
FastAPI server for the Graph RAG system
This allows Vercel to call your Graph RAG backend via HTTP
"""

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uvicorn
import os
import sys
from pathlib import Path

# Add current directory to Python path
sys.path.append(str(Path(__file__).parent))

from akriti_real_music_data import AKRITI_REAL_MUSIC_DATA
from music_graph_rag import MusicGraphRAG

app = FastAPI(title="Music Graph RAG API", version="1.0.0")

# Add CORS middleware for web app
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, restrict this to your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global RAG system (initialized lazily)
rag = None
rag_initialized = False

def initialize_rag():
    """Initialize the RAG system when first needed"""
    global rag, rag_initialized
    
    if rag_initialized:
        return rag
    
    try:
        print("Initializing Graph RAG system...")
        rag = MusicGraphRAG(demo_data=AKRITI_REAL_MUSIC_DATA, collection_prefix='akriti_web')
        rag.build_graph_structure(verbose=False)
        rag.create_embeddings_and_store()
        rag_initialized = True
        print("Graph RAG system ready!")
        return rag
    except Exception as e:
        print(f"Failed to initialize RAG system: {e}")
        rag_initialized = False
        return None

class QueryRequest(BaseModel):
    message: str

@app.get("/")
async def root():
    return {"message": "Music Graph RAG API is running"}

@app.post("/query")
async def query_music(request: QueryRequest):
    try:
        # Initialize RAG system if needed
        rag_system = initialize_rag()
        
        if rag_system is None:
            raise HTTPException(
                status_code=503,
                detail="Graph RAG system is not available. Check OpenAI API key and quota."
            )
        
        # Use your existing Graph RAG system
        result = rag_system.query_music_knowledge_with_llm_reranking(
            request.message, 
            compare_with_naive=True
        )
        
        # Clean up the result
        if result.startswith('[Graph RAG]'):
            result = result.replace('[Graph RAG]', '').strip()
        
        return {"response": result}
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"Error processing query: {e}")
        raise HTTPException(
            status_code=500, 
            detail=f"Error processing query: {str(e)}"
        )

@app.get("/health")
async def health_check():
    rag_system = initialize_rag()
    status = "healthy" if rag_system is not None else "unhealthy"
    return {
        "status": status, 
        "rag_system": "ready" if rag_system is not None else "failed",
        "openai_available": rag_system is not None
    }

if __name__ == "__main__":
    # For local development
    uvicorn.run(app, host="0.0.0.0", port=8000)
