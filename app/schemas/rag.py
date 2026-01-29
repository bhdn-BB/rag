from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any, Literal


class SourceInfoResponse(BaseModel):
    content: str = Field(..., description="Фрагмент тексту з документа")
    source: str = Field(..., description="Назва документа або URL")
    source_type: Literal["document", "url"] = Field(..., description="Тип джерела")

    # Для документів
    page: Optional[int] = Field(None, description="Номер сторінки (тільки для документів)")
    section: Optional[str] = Field(None, description="Назва розділу/секції (тільки для документів)")

    # Для обох типів
    score: Optional[float] = Field(None, description="Relevance score")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Додаткова metadata")

    def format_citation(self) -> str:
        if self.source_type == "url":
            return f"🔗 {self.source}"
        else:
            parts = [f"📄 {self.source}"]
            if self.page:
                parts.append(f"стор. {self.page}")
            if self.section:
                parts.append(f"розділ: {self.section}")
            return ", ".join(parts)


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
                        "source_type": "document",
                        "page": 12,
                        "section": "Introduction",
                        "score": 0.95,
                        "metadata": {}
                    },
                    {
                        "content": "According to the documentation...",
                        "source": "https://docs.example.com/ml",
                        "source_type": "url",
                        "page": None,
                        "section": None,
                        "score": 0.88,
                        "metadata": {}
                    }
                ],
                "query_rewritten": "machine learning definition",
                "rewrite_attempts": 0,
                "session_id": "user-123-chat-456",
                "error": None
            }
        }