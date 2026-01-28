from fastapi import FastAPI
from app.routers.vdb_crud import router as vdb_crud_router
from app.routers.agent import router as agent_router

app = FastAPI(title="Agentic RAG API")


app.include_router(router=vdb_crud_router, tags=["vdb_crud_router"])
app.include_router(agent_router, tags=["agent_router"])