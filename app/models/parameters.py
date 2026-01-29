from dataclasses import dataclass, field
from typing import List, Optional
import torch

from langchain_core.documents import Document


@dataclass
class BatchWorker:
    batch_size: int = 10
    num_workers: int = 2


@dataclass
class ChunkingParameters:
    chunk_size: int = 800
    chunk_overlap: int = 150
    h_separator: List[str] = field(default_factory=lambda: ["\n\n", "\n"])
    respect_sections: bool = True


@dataclass
class BiEncoderParams:
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    device: Optional[str] = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    max_length: int = 512
    normalize: bool = True


@dataclass
class CrossEncoderParams:
    model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"
    device: Optional[str] = field(
        default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu"
    )
    max_length: int = 512


# @dataclass
# class LLMSummarizerParams:
#     model_name: str = "gpt-4o-mini"
#     temperature: float = 0.0
#     max_input_tokens: int = 2000
#
#
# @dataclass
# class NLPSummarizerParams:
#     model_name: str = "facebook/bart-large-cnn"
#     max_length: int = 150
#     min_length: int = 40
#     num_beams: int = 4
#     length_penalty: float = 2.0
#     max_input_tokens: int = 2000


@dataclass
class SearchParameters:
    query: str
    top_k_retrieve: int = 50
    use_reranking: bool = False
    top_k_reranking: int = 5
    rerank_threshold: float = 0.2


@dataclass
class LLMParams:
    model_name: str = "gemini-2.5-flash"
    temperature: float = 0
    max_output_tokens: int = 3000


@dataclass
class SearchHit:
    document: Document
    score: Optional[float] = None