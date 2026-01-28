from abc import abstractmethod
from typing import Protocol

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

from app.models.parameters import BiEncoderParams, CrossEncoderParams


class BiEmbedder(Protocol):
    @abstractmethod
    def get_embedding(self, query: str) -> torch.Tensor:
        ...


class HFBiEmbedder(BiEmbedder):
    def __init__(self, params: BiEncoderParams):
        self.params = params
        self.device = params.device
        self.tokenizer = AutoTokenizer.from_pretrained(params.model_name)
        self.model = AutoModel.from_pretrained(params.model_name).to(self.device)
        self.model.eval()

    def get_embedding(self, query: str) -> torch.Tensor:
        encoded = self.tokenizer(
            query,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.params.max_length
        ).to(self.device)
        with torch.no_grad():
            out = self.model(**encoded)
            emb = out.last_hidden_state.mean(dim=1)
            if self.params.normalize:
                emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        return emb.squeeze(0)


class CrossEmbedder(Protocol):
    @abstractmethod
    def get_score(self, query: str, doc_text: str) -> float:
        ...


class HFCrossEncoder(CrossEmbedder):

    def __init__(self, params: CrossEncoderParams):
        self.params = params
        self.device = params.device
        self.tokenizer = AutoTokenizer.from_pretrained(params.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            params.model_name
        ).to(self.device)
        self.model.eval()

    def get_score(self, query: str, doc_text: str) -> float:
        inputs = self.tokenizer(
            query,
            doc_text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=self.params.max_length
        ).to(self.device)
        with torch.no_grad():
            logits = self.model(**inputs).logits
            score = torch.softmax(logits, dim=1)[:, 1]
        return score.item()