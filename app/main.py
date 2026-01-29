from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import logging

from app.routers.vdb_crud import router as vector_memory_router
from app.routers.agent import router as agent_router

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

load_dotenv()

app = FastAPI(
    title="Agentic RAG API",
    description="RAG system with LangGraph agent",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "agentic-rag-api"
    }

@app.get("/")
async def root():
    return {
        "message": "Agentic RAG API is running",
        "docs": "/docs",
        "health": "/health"
    }

app.include_router(router=vector_memory_router)
app.include_router(router=agent_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )