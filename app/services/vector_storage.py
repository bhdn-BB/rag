import logging
from typing import List, Optional, Dict, Any
from pathlib import Path
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings

from app.models.parameters import SearchParameters, SearchHit
from app.services.embedders import HFBiEmbedder, HFCrossEncoder

logger = logging.getLogger("VectorMemory")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


class HFEmbeddingWrapper(Embeddings):
    def __init__(self, embedder: HFBiEmbedder, batch_size: int = 16):
        self.embedder = embedder
        self.batch_size = batch_size

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        embeddings: List[List[float]] = []
        for i in tqdm(
            range(0, len(texts), self.batch_size),
            desc="Embedding documents",
        ):
            batch = texts[i : i + self.batch_size]
            batch_embeddings = [
                self.embedder.get_embedding(text).cpu().tolist()
                for text in batch
            ]
            embeddings.extend(batch_embeddings)
            logger.info(
                f"Embedded batch {i // self.batch_size + 1} "
                f"(size={len(batch)})"
            )
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding = self.embedder.get_embedding(text).cpu().tolist()
        logger.info(f"Embedded query: {text[:50]}...")
        return embedding


class VectorMemory:
    def __init__(
        self,
        bi_embedder: HFBiEmbedder,
        cross_encoder: Optional[HFCrossEncoder] = None,
        persist_path: str = "/app/data/chroma_index",
        collection_name: str = "documents",
    ):
        self.persist_path = persist_path
        self.cross_encoder = cross_encoder

        Path(self.persist_path).mkdir(parents=True, exist_ok=True)

        self._vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_path,
            embedding_function=HFEmbeddingWrapper(bi_embedder),
        )
        logger.info(f"VectorMemory initialized at {self.persist_path}")

    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            logger.warning("No documents to add")
            return

        self._vector_store.add_texts(
            texts=[d.page_content for d in docs],
            metadatas=[d.metadata for d in docs],
        )
        logger.info(f"Added {len(docs)} documents")

    def delete_documents(self, filter_metadata: Dict[str, Any]) -> None:
        self._vector_store.delete(where=filter_metadata)
        logger.info(f"Deleted documents with filter={filter_metadata}")

    def clear(self) -> None:
        self._vector_store.delete(where={})
        logger.warning("Vector store cleared")

    def _retrieve(self, query: str, top_k: int) -> List[Document]:
        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        return retriever.invoke(query)
    def _rerank(
        self,
        query: str,
        docs: List[Document],
        threshold: float,
    ) -> List[SearchHit]:
        if not self.cross_encoder:
            return [SearchHit(d) for d in docs]
        hits: List[SearchHit] = []
        for doc in docs:
            score = self.cross_encoder.get_score(query, doc.page_content)
            if score >= threshold:
                hits.append(SearchHit(doc, score))
        hits.sort(key=lambda x: x.score or 0, reverse=True)
        return hits

    def search(self, params: SearchParameters) -> List[SearchHit]:
        docs = self._retrieve(params.query, params.top_k_retrieve)
        if params.use_reranking:
            hits = self._rerank(
                params.query,
                docs,
                params.rerank_threshold,
            )
            return hits[: params.top_k_reranking]
        return [SearchHit(d) for d in docs]


    def get_stats(self) -> Dict[str, Any]:
        collection = self._vector_store.get()
        return {
            "status": "ready",
            "num_documents": len(collection["ids"]),
            "persist_path": self.persist_path,
            "has_cross_encoder": self.cross_encoder is not None,
        }