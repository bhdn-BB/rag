from fastapi import APIRouter, HTTPException
import logging

from app.graph.agent_rag import RAGAgent
from app.schemas.rag import (
    RAGQueryRequest,
    RAGQueryResponse,
    SourceInfoResponse,
)

logger = logging.getLogger("AgentRouter")

router = APIRouter(prefix="/agent", tags=["Agent"])

try:
    agent = RAGAgent(max_rewrite_attempts=1)
    logger.info("RAGAgent initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize RAGAgent: {str(e)}")
    raise


@router.post("/chat", response_model=RAGQueryResponse)
def chat(request: RAGQueryRequest):
    try:
        logger.info(f"Processing query: {request.query[:100]}...")
        result = agent.run(request.query)
        sources_list = result.get("sources", [])
        sources_response = [
            SourceInfoResponse(
                content=src.get("content", ""),
                source=src.get("source", "Unknown"),
                page=src.get("page"),
                section=src.get("section"),
                score=src.get("score"),
                metadata=src.get("metadata", {})
            )
            for src in sources_list
        ]
        logger.info(f"Response generated with {len(sources_response)} sources")
        return RAGQueryResponse(
            answer=result.get("answer", ""),
            sources=sources_response,
            query_rewritten=result.get("query"),
            rewrite_attempts=result.get("rewrite_attempts", 0),
            error=result.get("error")
        )
    except Exception as e:
        logger.error(f"Query failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to process query: {str(e)}"
        )


@router.post("/reset")
def reset():
    try:
        agent.reset()
        logger.info("Agent state reset")
        return {
            "status": "ok",
            "message": "Conversation reset successfully"
        }
    except Exception as e:
        logger.error(f"Reset failed: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Reset failed: {str(e)}"
        )


@router.get("/status")
def status():
    try:
        return {
            "status": "ready",
            "max_rewrite_attempts": agent.max_rewrite_attempts,
            "message": "Agent is ready to process queries"
        }
    except Exception as e:
        logger.error(f"Status check failed: {str(e)}")
        return {
            "status": "error",
            "error": str(e)
        }