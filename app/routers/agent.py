from fastapi import APIRouter, HTTPException
import logging
from typing import Dict, Optional

from app.graph.agent_rag import RAGAgent
from app.schemas.rag import RAGQueryRequest, RAGQueryResponse, SourceInfoResponse

logger = logging.getLogger("AgentRouter")

router = APIRouter(prefix="/agent", tags=["Agent"])

agents: Dict[str, RAGAgent] = {}

default_agent = RAGAgent(max_rewrite_attempts=1)
logger.info("Default RAGAgent initialized")


def get_or_create_agent(session_id: Optional[str] = None) -> RAGAgent:

    if session_id is None:
        return default_agent

    if session_id not in agents:
        logger.info(f"Creating new agent for session: {session_id}")
        agents[session_id] = RAGAgent(max_rewrite_attempts=1)

    return agents[session_id]


def convert_sources_to_response(sources: list) -> list:
    result = []
    for src in sources:
        result.append(
            SourceInfoResponse(
                content=src.get("content", ""),
                source=src.get("source", "Unknown"),
                page=src.get("page"),
                section=src.get("section"),
                score=src.get("score"),
                metadata=src.get("metadata", {})
            )
        )

    return result


@router.post("/chat", response_model=RAGQueryResponse)
def chat(request: RAGQueryRequest):
    try:
        session_id = request.session_id
        logger.info(f"Processing query: {request.query[:100]}... (session: {session_id or 'none'})")

        agent = get_or_create_agent(session_id)

        result = agent.run(request.query)

        answer = result.get("answer", "")
        sources_list = result.get("sources", [])

        sources_response = convert_sources_to_response(sources_list)

        if sources_response:
            logger.info(f"Answer with {len(sources_response)} sources (session: {session_id or 'default'})")
        else:
            logger.info(f"No sources, fallback answer (session: {session_id or 'default'})")

        return RAGQueryResponse(
            answer=answer,
            sources=sources_response,
            query_rewritten=result.get("query"),
            rewrite_attempts=result.get("rewrite_attempts", 0),
            session_id=session_id,
            error=None
        )

    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )