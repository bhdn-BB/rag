from typing import List, Optional, Dict, Any
from pydantic import BaseModel


class SearchRequest(BaseModel):
    query: str
    top_k_retrieve: int = 50
    use_reranking: bool = False
    top_k_reranking: int = 5
    rerank_threshold: float = 0.2


class SearchResultItem(BaseModel):
    score: Optional[float]
    content: str
    metadata: Dict[str, Any]


class SearchResponse(BaseModel):
    results: List[SearchResultItem]


class DeleteByMetadataRequest(BaseModel):
    filter_metadata: Dict[str, Any]