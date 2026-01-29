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


def determine_source_type(source: str, metadata: dict) -> str:
    if metadata.get("source_type"):
        return metadata["source_type"]
    if source.startswith(("http://", "https://", "www.")):
        return "url"
    if metadata.get("source") and isinstance(metadata["source"], str):
        if metadata["source"].startswith(("http://", "https://", "www.")):
            return "url"
    return "document"


def convert_sources_to_response(sources: list) -> list:

    result = []

    for src in sources:
        if hasattr(src, 'document'):
            content = src.document.page_content
            metadata = src.document.metadata
            score = src.score
            source_name = metadata.get("source", "Unknown")
            page = metadata.get("page")
            section = metadata.get("section")
        elif isinstance(src, dict):
            content = src.get("content", "")
            metadata = src.get("metadata", {})
            score = src.get("score")
            source_name = src.get("source", metadata.get("source", "Unknown"))
            page = src.get("page", metadata.get("page"))
            section = src.get("section", metadata.get("section"))
        else:
            logger.warning(f"Unknown source type: {type(src)}")
            continue
        source_type = determine_source_type(source_name, metadata)
        if source_type == "url":
            result.append(
                SourceInfoResponse(
                    content=content,
                    source=source_name,
                    source_type="url",
                    page=None,
                    section=None,
                    score=score,
                    metadata=metadata
                )
            )
        else:
            result.append(
                SourceInfoResponse(
                    content=content,
                    source=source_name,
                    source_type="document",
                    page=page,
                    section=section,
                    score=score,
                    metadata=metadata
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
            url_count = sum(1 for s in sources_response if s.source_type == "url")
            doc_count = sum(1 for s in sources_response if s.source_type == "document")
            logger.info(f"Answer with {len(sources_response)} sources ({url_count} URLs, {doc_count} docs)")
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
        logger.error(f"Query failed: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.delete("/session/{session_id}")
def delete_session(session_id: str):
    if session_id in agents:
        del agents[session_id]
        logger.info(f"Deleted session: {session_id}")
        return {"status": "ok", "message": f"Session {session_id} deleted"}
    return {"status": "not_found", "message": f"Session {session_id} not found"}


@router.get("/sessions")
def list_sessions():
    return {
        "active_sessions": list(agents.keys()),
        "count": len(agents)
    }