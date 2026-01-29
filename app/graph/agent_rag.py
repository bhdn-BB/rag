import logging
from typing import List, Literal, Optional

from langchain_core.documents import Document
from langgraph.graph import END, START, StateGraph

from app.graph.llm_client import LLMClient
from app.graph.nodes.fallback_node import FallbackNode
from app.graph.nodes.generate_node import GenerateNode
from app.graph.nodes.grade_node import GradeNode
from app.graph.nodes.input_node import InputNode
from app.graph.nodes.retrieve_node import RetrieveNode
from app.graph.nodes.rewrite_node import RewriteNode
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


RouteResult = Literal["rewrite", "fallback", "__end__"]

def route_after_grade(
    state: GraphState,
    max_attempts: int,
) -> RouteResult:
    answer: str = state.get("answer", "")

    if answer.strip():
        logger.info("Answer found, finishing graph execution")
        return "__end__"

    if state["rewrite_attempts"] < max_attempts:
        logger.info(
            "Answer not found, retry rewrite | attempt=%d",
            state["rewrite_attempts"] + 1,
        )
        return "rewrite"

    logger.info("Rewrite attempts exhausted, fallback")
    return "fallback"


class RAGAgent:
    def __init__(self, max_rewrite_attempts: int = 1) -> None:
        self.llm: LLMClient = LLMClient()

        self.vector_memory: VectorMemory = VectorMemory(
            bi_embedder=HFBiEmbedder(BiEncoderParams()),
            cross_encoder=HFCrossEncoder(CrossEncoderParams()),
        )

        self.max_rewrite_attempts: int = max_rewrite_attempts
        self.graph = self._build_graph()

        logger.info(
            "RAGAgent initialized | max_rewrite_attempts=%d",
            self.max_rewrite_attempts,
        )


    def _build_graph(self):
        graph: StateGraph = StateGraph(GraphState)

        graph.add_node("input", InputNode())
        graph.add_node("rewrite", RewriteNode(self.llm))
        graph.add_node(
            "retrieve",
            RetrieveNode(self.vector_memory),
            max_attempts=1,
        )
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
            lambda state: route_after_grade(
                state,
                self.max_rewrite_attempts,
            ),
        )

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
        }

        logger.info("Graph execution started | query=%s", query[:50])

        return self.graph.invoke(initial_state)