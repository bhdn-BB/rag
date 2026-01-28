from typing import Dict
from app.services.vector_storage import VectorMemory
from app.models.parameters import SearchParameters

class RetrieveNode:
    def __init__(self, vector_memory: VectorMemory):
        self.vector_memory = vector_memory

    def __call__(self, state: Dict) -> Dict:
        query = state.get("query", "")
        params = SearchParameters(query=query, top_k_retrieve=5)
        docs = self.vector_memory.search(params)
        state["docs"] = [doc[1] if isinstance(doc, tuple) else doc for doc in docs]
        return state
