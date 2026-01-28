from typing import Dict
from app.graph.llm_client import LLMClient

class RewriteNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def __call__(self, state: Dict) -> Dict:
        query = state.get("query", "")
        prompt = f"Перефразуй цей запит для покращення пошуку: '{query}'"
        rewritten = self.llm_client.generate(prompt)
        state["query"] = rewritten
        return state
