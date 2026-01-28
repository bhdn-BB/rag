from typing import Optional

from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k_retrieve: int = 50
    use_reranking: bool = False
    top_k_reranking: int = 5
    rerank_threshold: float = 0.2


class SearchResultItem(BaseModel):
    score: Optional[float] = None
    content: str
    metadata: dict


class SearchResponse(BaseModel):
    results: List[SearchResultItem]