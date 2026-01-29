# app/graph/agent_rag.py
from typing import List, Optional, Literal
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
import logging

from app.graph.nodes.fallback_node import FallbackNode
from app.graph.nodes.generate_node import GenerateNode
from app.graph.nodes.grade_node import GradeNode
from app.graph.nodes.input_node import InputNode
from app.graph.nodes.retrieve_node import RetrieveNode
from app.graph.nodes.rewrite_node import RewriteNode
from app.graph.state_model import GraphState
from app.services.vector_storage import VectorMemory
from app.graph.llm_client import LLMClient
from app.models.parameters import BiEncoderParams, CrossEncoderParams
from app.services.embedders import HFBiEmbedder, HFCrossEncoder

logger = logging.getLogger("RAGAgent")


def route_after_grade(state: GraphState, max_attempts: int) -> Literal["rewrite", "fallback", "__end__"]:

    if state.get("answer") and state["answer"].strip():
        logger.info("Відповідь знайдена, завершуємо")
        return "__end__"

    if state["rewrite_attempts"] < max_attempts:
        logger.info(f"Відповідь не знайдена, retry (спроба {state['rewrite_attempts'] + 1})")
        return "rewrite"

    logger.info("Вичерпано спроби, fallback")
    return "fallback"


class RAGAgent:
    def __init__(self, max_rewrite_attempts: int = 1):
        self.llm = LLMClient()
        self.vector_memory = VectorMemory(
            HFBiEmbedder(BiEncoderParams()),
            HFCrossEncoder(CrossEncoderParams())
        )
        self.max_rewrite_attempts = max_rewrite_attempts
        self.graph = self._build_graph()

    def _build_graph(self):
        graph = StateGraph(GraphState)

        graph.add_node("input", InputNode())
        graph.add_node("rewrite", RewriteNode(self.llm))
        graph.add_node("retrieve", RetrieveNode(self.vector_memory))
        graph.add_node("generate", GenerateNode(self.llm))
        graph.add_node("grade", GradeNode())
        graph.add_node("fallback", FallbackNode())

        graph.add_edge(START, "input")
        graph.add_edge("input", "rewrite")
        graph.add_edge("rewrite", "retrieve")
        graph.add_edge("retrieve", "generate")
        graph.add_edge("generate", "grade")

        graph.add_conditional_edges(
            "grade",
            lambda state: route_after_grade(state, self.max_rewrite_attempts)
        )
        graph.add_edge("fallback", END)
        return graph.compile()

    def run(self, query: str, docs: Optional[List[Document]] = None) -> dict:
        state: GraphState = {
            "input_query": query,
            "query": "",
            "docs": docs or [],
            "answer": "",
            "sources": [],
            "rewrite_attempts": 0,
            "messages": []
        }
        return self.graph.invoke(state)