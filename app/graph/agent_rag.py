from langgraph.graph.state import StateGraph, CompiledStateGraph
from langgraph.checkpoint.memory import InMemorySaver

from app.graph.nodes.input_node import InputNode
from app.graph.nodes.retrieve_node import RetrieveNode
from app.graph.nodes.grade_node import GradeNode
from app.graph.nodes.rewrite_node import RewriteNode
from app.graph.nodes.generate_node import GenerateNode
from app.graph.llm_client import LLMClient, LLMParams
from app.services.vector_storage import VectorMemory


class AgenticRAG:
    def __init__(self, vector_memory: VectorMemory, llm_params: LLMParams):
        self.vector_memory = vector_memory
        self.llm_client = LLMClient(params=llm_params)

        self.input_node = InputNode()
        self.retrieve_node = RetrieveNode(vector_memory)
        self.grade_node = GradeNode()
        self.rewrite_node = RewriteNode(self.llm_client)
        self.generate_node = GenerateNode(self.llm_client)

        self.graph = StateGraph(initial_state={})
        self.memory_saver = InMemorySaver()
        self._build_graph()
        self.compiled: CompiledStateGraph = self.graph.compile(memory=self.memory_saver)

    def _build_graph(self):
        self.graph.add_node("input_node", self.input_node)
        self.graph.add_node("retrieve_node", self.retrieve_node)
        self.graph.add_node("grade_node", self.grade_node)
        self.graph.add_node("rewrite_node", self.rewrite_node)
        self.graph.add_node("generate_node", self.generate_node)

        self.graph.add_edge("input_node", "retrieve_node")
        self.graph.add_edge("retrieve_node", "grade_node")
        self.graph.add_edge(
            "grade_node",
            "generate_node",
            condition=lambda state: len(state.get("docs", [])) >= 2
        )
        self.graph.add_edge(
            "grade_node",
            "rewrite_node",
            condition=lambda state: len(state.get("docs", [])) < 2
        )
        self.graph.add_edge("rewrite_node", "retrieve_node")

    def run(self, query: str, session_id: str = "default") -> str:
        state = {"input_query": query, "session_id": session_id}
        result_state = self.compiled.invoke(state)
        return result_state.get("answer", "")

    def stream(self, query: str, session_id: str = "default", stream_mode: list = ["updates"]):
        state = {"input_query": query, "session_id": session_id}
        for update in self.compiled.stream(state, stream_mode=stream_mode):
            yield update
