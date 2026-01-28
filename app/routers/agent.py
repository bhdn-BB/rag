from fastapi import APIRouter, Query
from fastapi.responses import StreamingResponse
import json

from app.graph.agent_rag import AgenticRAG
from app.models.parameters import BiEncoderParams, CrossEncoderParams, LLMParams
from app.services.vector_storage import VectorMemory

router = APIRouter(prefix="/agent", tags=["Agentic RAG"])

vector_memory = VectorMemory(bi_embedder=BiEncoderParams(), cross_encoder=CrossEncoderParams())
llm_params = LLMParams(model_name="gemini-2.5-flash", temperature=0, max_output_tokens=512)
agent = AgenticRAG(vector_memory, llm_params)


@router.get("/run")
def run_agent(query: str = Query(...)):
    result = agent.run(query)
    return {
        "answer": result.get("answer", ""),
        "sources": result.get("sources", [])
    }

@router.get("/stream")
def stream_agent(query: str = Query(...)):
    def event_generator():
        for update in agent.stream(query):
            yield json.dumps({
                "answer": update.get("answer", ""),
                "sources": update.get("sources", [])
            }) + "\n"
    return StreamingResponse(event_generator(), media_type="application/json")