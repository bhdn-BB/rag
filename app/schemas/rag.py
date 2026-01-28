from pydantic import BaseModel, Field
from typing import List, Optional, Dict, Any


class SourceInfoResponse(BaseModel):
    content: str = Field(..., description="Фрагмент тексту з документа")
    source: str = Field(..., description="Назва документа або URL")
    page: Optional[int] = Field(None, description="Номер сторінки (якщо доступно)")
    section: Optional[str] = Field(None, description="Назва розділу/секції")
    score: Optional[float] = Field(None, description="Relevance score від reranker")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Додаткова metadata")

    class Config:
        json_schema_extra = {
            "example": {
                "content": "Artificial intelligence is the simulation of human intelligence...",
                "source": "AI_Introduction.pdf",
                "page": 5,
                "section": "Chapter 1: Basics",
                "score": 0.92,
                "metadata": {
                    "author": "John Doe",
                    "date": "2024-01-15"
                }
            }
        }


class RAGQueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Питання користувача")

    class Config:
        json_schema_extra = {
            "example": {
                "query": "What is machine learning?"
            }
        }


class RAGQueryResponse(BaseModel):
    answer: str = Field(..., description="Згенерована відповідь")
    sources: List[SourceInfoResponse] = Field(
        default_factory=list,
        description="Список джерел з детальною інформацією"
    )
    query_rewritten: Optional[str] = Field(None, description="Переписаний запит (якщо був rewrite)")
    rewrite_attempts: int = Field(0, description="Кількість спроб переформулювання")
    error: Optional[str] = Field(None, description="Повідомлення про помилку")

    class Config:
        json_schema_extra = {
            "example": {
                "answer": "Machine learning is a subset of artificial intelligence...",
                "sources": [
                    {
                        "content": "Machine learning allows computers to learn...",
                        "source": "ML_Guide.pdf",
                        "page": 12,
                        "section": "Introduction to ML",
                        "score": 0.95,
                        "metadata": {}
                    }
                ],
                "query_rewritten": "definition of machine learning algorithms",
                "rewrite_attempts": 1,
                "error": None
            }
        }