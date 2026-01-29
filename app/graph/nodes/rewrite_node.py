from app.graph.llm_client import LLMClient
from app.graph.state_model import GraphState


class RewriteNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def __call__(self, state: GraphState) -> dict:
        query = state["query"]
        if state["rewrite_attempts"] == 0:
            return {"query": query}
        prompt = f"Перефразуй цей запит для покращення пошуку: '{query}'"
        rewritten = self.llm_client.generate(prompt)
        return {
            "query": rewritten,
            "rewrite_attempts": state["rewrite_attempts"] + 1
        }