from typing import Dict
from app.services.vector_storage import VectorMemory
from app.models.parameters import SearchParameters
import logging

logger = logging.getLogger("RetrieveNode")

class RetrieveNode:
    def __init__(self, vector_memory: VectorMemory):
        self.vector_memory = vector_memory

    def __call__(self, state: Dict) -> Dict:
        search_params = SearchParameters(query=state["query"])
        state["docs"] = self.vector_memory.search(search_params)
        logger.info(f"Retrieved {len(state['docs'])} documents")
        return state
