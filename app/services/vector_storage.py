import logging
from typing import List, Optional, Dict, Any, Tuple
from pathlib import Path
from tqdm import tqdm

from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_core.embeddings import Embeddings

from app.models.parameters import SearchParameters
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
        persist_path: Optional[str] = None,
        batch_size: int = 16,
    ):
        self.bi_embedder = bi_embedder
        self.cross_encoder = cross_encoder
        self.persist_path = persist_path
        self.batch_size = batch_size
        self._vector_store: Optional[Chroma] = None

    def build_index(self, docs: List[Document]) -> None:
        if not docs:
            raise ValueError("Cannot build index with empty document list")
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        logger.info(f"Building vector index for {len(docs)} documents")
        self._vector_store = Chroma.from_texts(
            texts=texts,
            metadatas=metadatas,
            embedding=HFEmbeddingWrapper(
                self.bi_embedder,
                batch_size=self.batch_size,
            ),
            persist_directory=self.persist_path,
        )
        if self.persist_path:
            Path(self.persist_path).mkdir(parents=True, exist_ok=True)
            self._vector_store.persist()
            logger.info(f"Index persisted at {self.persist_path}")

    def load_index(self) -> None:
        if not self.persist_path:
            raise ValueError("persist_path must be set to load index")
        if not Path(self.persist_path).exists():
            raise ValueError(
                f"Index directory does not exist: {self.persist_path}"
            )
        logger.info(f"Loading index from {self.persist_path}")
        self._vector_store = Chroma(
            persist_directory=self.persist_path,
            embedding_function=HFEmbeddingWrapper(
                self.bi_embedder,
                batch_size=self.batch_size,
            ),
        )
        logger.info("Index loaded successfully")


    def add_documents(self, docs: List[Document]) -> None:
        if not docs:
            logger.warning("No documents to add")
            return
        if self._vector_store is None:
            logger.info("Vector store empty, building new index")
            self.build_index(docs)
            return
        texts = [doc.page_content for doc in docs]
        metadatas = [doc.metadata for doc in docs]
        self._vector_store.add_texts(
            texts=texts,
            metadatas=metadatas,
        )
        logger.info(f"Added {len(docs)} documents to vector store")
        if self.persist_path:
            self._vector_store.persist()
            logger.info("Vector store persisted")

    def delete_documents(self, filter_metadata: Dict[str, Any]) -> None:
        if self._vector_store is None:
            raise ValueError("Vector store is empty")
        before = len(self._vector_store.get()["ids"])
        self._vector_store.delete(where=filter_metadata)
        after = len(self._vector_store.get()["ids"])
        logger.info(
            f"Deleted {before - after} documents "
            f"with filter {filter_metadata}"
        )
        if self.persist_path:
            self._vector_store.persist()

    def clear(self) -> None:
        if self._vector_store is None:
            logger.warning("Vector store already empty")
            return
        self._vector_store.delete(where={})
        logger.info("Vector store cleared")
        if self.persist_path:
            self._vector_store.persist()


    def retrieve(self, query: str, top_k: int) -> List[Document]:
        if self._vector_store is None:
            raise ValueError("Vector store is empty")
        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": top_k}
        )
        docs = retriever.get_relevant_documents(query)
        logger.info(f"Retrieved {len(docs)} documents")
        return docs

    def rerank(
        self,
        query: str,
        docs: List[Document],
        threshold: float,
    ) -> List[Tuple[float, Document]]:
        if not self.cross_encoder:
            raise ValueError("Cross-encoder not provided")
        if not docs:
            return []
        scored_docs: List[Tuple[float, Document]] = []
        logger.info(
            f"Reranking {len(docs)} documents "
            f"(threshold={threshold})"
        )
        for doc in docs:
            score = self.cross_encoder.get_score(
                query,
                doc.page_content,
            )
            if score >= threshold:
                scored_docs.append((score, doc))
        scored_docs.sort(key=lambda x: x[0], reverse=True)
        logger.info(
            f"{len(scored_docs)} documents passed reranking threshold"
        )
        return scored_docs

    def search(self, params: SearchParameters):
        if params.use_reranking:
            candidates = self.retrieve(params.query, params.top_k_retrieve)
            reranked = self.rerank(params.query, candidates, params.rerank_threshold)
            return reranked[:params.top_k_retrieve]
        return self.retrieve(params.query, params.top_k_retrieve)

    def get_stats(self) -> Dict[str, Any]:
        collection = self._vector_store.get()
        return {
            "status": "ready",
            "num_documents": len(collection["ids"]),
            "persist_path": self.persist_path,
            "has_cross_encoder": self.cross_encoder is not None,
        }