from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SourceInfoResponse(BaseModel):
    content: str = Field(..., description="Фрагмент тексту з документа")
    source: str = Field(..., description="Назва документа або URL")
    page: Optional[int] = Field(None, description="Номер сторінки")
    section: Optional[str] = Field(None, description="Назва розділу/секції")
    score: Optional[float] = Field(None, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Додаткова metadata")


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Питання користувача")
    session_id: Optional[str] = Field(None, description="ID сесії для контексту (опціонально)")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?",
                "session_id": "user-123-chat-456"
            }
        }


class RAGQueryResponse(BaseModel):
    answer: str = Field(..., description="Згенерована відповідь")
    sources: List[SourceInfoResponse] = Field(
        default_factory=list,
        description="Список джерел (може бути порожнім якщо нічого не знайдено)"
    )
    query_rewritten: Optional[str] = Field(None, description="Переписаний запит")
    rewrite_attempts: int = Field(0, description="Кількість спроб переформулювання")
    session_id: Optional[str] = Field(None, description="ID сесії")
    error: Optional[str] = Field(None, description="Помилка")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is...",
                "sources": [
                    {
                        "content": "ML allows computers to learn...",
                        "source": "ML_Guide.pdf",
                        "page": 12,
                        "section": "Introduction",
                        "score": 0.95,
                        "metadata": {}
                    }
                ],
                "query_rewritten": "machine learning definition",
                "rewrite_attempts": 0,
                "session_id": "user-123-chat-456",
                "error": None
            }
        }