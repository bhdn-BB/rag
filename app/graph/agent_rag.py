# app/graph/agent_rag.py
from typing import List, Optional, Literal, Any
from langgraph.graph import StateGraph, START, END
from langchain_core.documents import Document
import logging

from langgraph.graph.state import CompiledStateGraph

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


class RAGAgent:

    def __init__(self, max_rewrite_attempts: int = 1):
        self.llm = LLMClient()
        self.vector_memory = VectorMemory(
            HFBiEmbedder(BiEncoderParams()),
            HFCrossEncoder(CrossEncoderParams())
        )
        self.max_rewrite_attempts = max_rewrite_attempts
        self.graph = self._build_graph()

    def _build_graph(self) -> CompiledStateGraph[Any, Any, Any, Any]:
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

        def route_after_grade(state: GraphState) -> Literal["rewrite", "fallback", "__end__"]:
            if state["answer"]:
                return "__end__"
            if state["rewrite_attempts"] < self.max_rewrite_attempts and state["docs"]:
                return "rewrite"
            return "fallback"

        graph.add_conditional_edges(
            "grade",
            route_after_grade
        )

        graph.add_edge("fallback", END)

        return graph.compile()

    def run(self, query: str, docs: Optional[List[Document]] = None) -> dict:
        state: GraphState = {
            "input_query": query,
            "docs": docs,
            "query": "",
            "answer": "",
            "sources": [],
            "docs_count": 0,
            "rewrite_attempts": 0,
            "messages": []
        }
        result = self.graph.invoke(state)
        return result

    def reset(self):
        logger.info("RAGAgent state has been reset")