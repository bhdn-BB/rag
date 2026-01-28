from typing import List
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever

class BM25Search:
    def __init__(self, docs: List[Document], k: int = 5):
        self.retriever = BM25Retriever.from_documents(
            docs,
            k=k
        )
    def search(self, query: str) -> List[Document]:
        results = self.retriever.invoke(query)
        return results
