import logging
from typing import List, Literal, Optional

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.graph.llm_client import LLMClient
from app.graph.nodes.fallback_node import FallbackNode
from app.graph.nodes.generate_node import GenerateNode
from app.graph.nodes.grade_node import GradeNode
from app.graph.nodes.input_node import InputNode
from app.graph.nodes.query_analysis import QueryAnalysisNode
from app.graph.nodes.retrieve_node import RetrieveNode
from app.graph.nodes.rewrite_node import RewriteQueryNode
from app.graph.state_model import GraphState
from app.models.parameters import BiEncoderParams, CrossEncoderParams
from app.services.embedders import HFBiEmbedder, HFCrossEncoder
from app.services.vector_storage import VectorMemory

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: logging.Logger = logging.getLogger(__name__)



class RAGAgent:
    def __init__(
            self,
            max_rewrite_attempts: int = 1,
    ) -> None:
        self.llm: LLMClient = LLMClient()
        self.max_rewrite_attempts: int = max_rewrite_attempts

        self.vector_memory: VectorMemory = VectorMemory(
            bi_embedder=HFBiEmbedder(BiEncoderParams()),
            cross_encoder=HFCrossEncoder(CrossEncoderParams()),
        )

        self.graph = self._build_graph()
        logger.info(
            "RAGAgent ініціалізовано | max_rewrite_attempts=%d",
            self.max_rewrite_attempts,
        )

    def _route_after_query_analysis(self, state: GraphState) -> str:
        need_external = state.get("need_external_info", True)
        if need_external:
            logger.info("Потрібна зовнішня інформація -> RAG")
            return "retrieve"
        else:
            logger.info("Зовнішня інформація не потрібна -> пряма генерація")
            return "generate"

    def _route_after_grade(self, state: GraphState) -> str:
        if state.get("enough_data", False):
            logger.info("Достатньо інформації в документах -> генерація")
            return "generate"
        else:
            rewrite_attempts = state.get("rewrite_attempts", 0)
            if rewrite_attempts < self.max_rewrite_attempts:
                logger.info(f"Недостатньо інформації -> переписування (спроба {rewrite_attempts + 1}/{self.max_rewrite_attempts})")
                return "rewrite"
            else:
                logger.warning("Недостатньо інформації і спроби вичерпано -> fallback")
                return "fallback"

    def _build_graph(self):

        graph = StateGraph(GraphState)
        graph.add_node("input", InputNode())
        graph.add_node("query_analysis", QueryAnalysisNode(self.llm))
        graph.add_node("retrieve", RetrieveNode(self.vector_memory))
        graph.add_node("grade", GradeNode(self.llm))
        graph.add_node("rewrite", RewriteQueryNode(self.llm))
        graph.add_node("generate", GenerateNode(self.llm))
        graph.add_node("fallback", FallbackNode())

        graph.add_edge(START, "input")
        graph.add_edge("input", "query_analysis")
        graph.add_conditional_edges(
            "query_analysis",
            self._route_after_query_analysis,
            {
                "retrieve": "retrieve",
                "generate": "generate",
            }
        )

        graph.add_edge("retrieve", "grade")

        graph.add_conditional_edges(
            "grade",
            self._route_after_grade,
            {
                "generate": "generate",
                "rewrite": "rewrite",
                "fallback": "fallback",
            }
        )
        graph.add_edge("rewrite", "query_analysis")
        graph.add_edge("generate", END)
        graph.add_edge("fallback", END)
        return graph.compile()

    def run(
            self,
            query: str,
            docs: Optional[List[Document]] = None,
    ) -> dict:
        initial_state: GraphState = {
            "input_query": query,
            "query": "",
            "docs": docs or [],
            "answer": "",
            "sources": [],
            "rewrite_attempts": 0,
            "messages": [],
            "need_external_info": True,
            "enough_data": False,
        }
        logger.info(f"Початок виконання графа | запит: {query[:100]}")
        result = self.graph.invoke(initial_state)
        logger.info(f"Відповідь: {result.get('answer', '')[:100]}...")
        logger.info(f"Джерел: {len(result.get('sources', []))}")
        return result