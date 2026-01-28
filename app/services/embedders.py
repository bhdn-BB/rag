from abc import abstractmethod
from typing import Protocol
import logging

import torch
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoModelForSequenceClassification,
)

from app.models.parameters import BiEncoderParams, CrossEncoderParams

logger = logging.getLogger("Embedders")


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
        logger.info(f"BiEmbedder loaded: {params.model_name} on {self.device}")

    def get_embedding(self, query: str) -> torch.Tensor:
        try:
            encoded = self.tokenizer(
                query,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.params.max_length,
                return_attention_mask=True,
            ).to(self.device)

            with torch.no_grad():
                out = self.model(**encoded)
                attention_mask = encoded['attention_mask']
                token_embeddings = out.last_hidden_state

                input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
                sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, 1)
                sum_mask = torch.clamp(input_mask_expanded.sum(1), min=1e-9)
                emb = sum_embeddings / sum_mask

                if self.params.normalize:
                    emb = torch.nn.functional.normalize(emb, p=2, dim=1)

            return emb.squeeze(0)

        except Exception as e:
            logger.error(f"Embedding error for text (len={len(query)}): {str(e)}")
            # Fallback: повертаємо zero vector
            return torch.zeros(self.model.config.hidden_size, device=self.device)


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
        logger.info(f"CrossEncoder loaded: {params.model_name} on {self.device}")

    def get_score(self, query: str, doc_text: str) -> float:
        try:
            inputs = self.tokenizer(
                query,
                doc_text,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=self.params.max_length,
                return_attention_mask=True,
            ).to(self.device)

            with torch.no_grad():
                logits = self.model(**inputs).logits
                if logits.shape[-1] == 2:
                    score = torch.softmax(logits, dim=1)[:, 1]
                else:
                    score = torch.sigmoid(logits)
            return float(score.item())
        except Exception as e:
            logger.error(f"Scoring error: {str(e)}")
            return 0.0