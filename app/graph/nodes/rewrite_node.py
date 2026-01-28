from typing import Dict
from app.graph.llm_client import LLMClient
import logging

logger = logging.getLogger("RewriteNode")
logger.setLevel(logging.INFO)


class RewriteNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def __call__(self, state: Dict) -> Dict:
        query = state["query"]
        prompt = f"Перефразуй цей запит для покращення пошуку: '{query}'"
        state["query"] = self.llm_client.generate(prompt)
        logger.info(f"Query rewritten: {state['query']}")
        return state