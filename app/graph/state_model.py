from typing import List, Annotated, Dict, Any

from langchain_core.messages import AnyMessage
from langgraph.graph import add_messages
from langchain_core.documents import Document
from typing_extensions import TypedDict


class SourceInfo(TypedDict):
    content: str
    source: str
    page: int | None
    section: str | None
    score: float | None
    metadata: Dict[str, Any]


class GraphState(TypedDict):
    input_query: str
    query: str
    docs: List[Document]
    answer: str
    sources: List[SourceInfo]
    rewrite_attempts: Annotated[int, lambda x, y: x + y]
    messages: Annotated[List[AnyMessage], add_messages]
    enough_data: bool