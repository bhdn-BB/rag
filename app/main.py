from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.routers.vdb_crud import router as vdb_crud_router
from app.routers.agent import router as agent_router

app = FastAPI(title="Agentic RAG API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(router=vdb_crud_router, tags=["vdb_crud_router"])
app.include_router(agent_router, tags=["agent_router"])