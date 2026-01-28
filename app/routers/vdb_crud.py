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
        if not texts:
            logger.warning("No texts to embed")
            return []
        embeddings: List[List[float]] = []
        failed_count = 0
        for i in tqdm(
                range(0, len(texts), self.batch_size),
                desc="Embedding documents",
        ):
            batch = texts[i: i + self.batch_size]
            batch_embeddings = []
            for text in batch:
                try:
                    if not text or not text.strip():
                        logger.warning(f"Empty text at index {i}, using minimal text")
                        text = " "
                    max_chars = self.embedder.params.max_length * 4
                    if len(text) > max_chars:
                        logger.warning(f"Text too long ({len(text)} chars), truncating to {max_chars}")
                        text = text[:max_chars]
                    emb = self.embedder.get_embedding(text)
                    batch_embeddings.append(emb.cpu().tolist())
                except Exception as e:
                    logger.error(f"Failed to embed text: {str(e)}")
                    failed_count += 1
                    zero_vec = [0.0] * self.embedder.model.config.hidden_size
                    batch_embeddings.append(zero_vec)
            embeddings.extend(batch_embeddings)
            logger.info(
                f"Embedded batch {i // self.batch_size + 1} "
                f"(size={len(batch)}, failed={failed_count})"
            )
        if failed_count > 0:
            logger.warning(f"Total failed embeddings: {failed_count}/{len(texts)}")
        return embeddings

    def embed_query(self, text: str) -> List[float]:
        try:
            if not text or not text.strip():
                logger.warning("Empty query text")
                text = " "
            max_chars = self.embedder.params.max_length * 4
            if len(text) > max_chars:
                logger.warning(f"Query too long ({len(text)} chars), truncating")
                text = text[:max_chars]
            embedding = self.embedder.get_embedding(text).cpu().tolist()
            logger.info(f"Embedded query: {text[:50]}...")
            return embedding
        except Exception as e:
            logger.error(f"Failed to embed query: {str(e)}")
            return [0.0] * self.embedder.model.config.hidden_size


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
        valid_docs = []
        for doc in docs:
            if doc.page_content and doc.page_content.strip():
                max_chars = 8192
                if len(doc.page_content) > max_chars:
                    logger.warning(f"Document too long ({len(doc.page_content)} chars), truncating")
                    doc.page_content = doc.page_content[:max_chars]
                valid_docs.append(doc)
            else:
                logger.warning("Skipping empty document")

        if not valid_docs:
            logger.warning("No valid documents to add after filtering")
            return

        try:
            self._vector_store.add_texts(
                texts=[d.page_content for d in valid_docs],
                metadatas=[d.metadata for d in valid_docs],
            )
            logger.info(f"Added {len(valid_docs)} documents (filtered {len(docs) - len(valid_docs)})")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def delete_documents(self, filter_metadata: Dict[str, Any]) -> None:
        try:
            self._vector_store.delete(where=filter_metadata)
            logger.info(f"Deleted documents with filter={filter_metadata}")
        except Exception as e:
            logger.error(f"Failed to delete documents: {str(e)}")
            raise

    def clear(self) -> None:
        try:
            self._vector_store.delete(where={})
            logger.warning("Vector store cleared")
        except Exception as e:
            logger.error(f"Failed to clear vector store: {str(e)}")
            raise

    def _retrieve(self, query: str, top_k: int) -> List[Document]:
        if not query or not query.strip():
            logger.warning("Empty query")
            return []
        try:
            retriever = self._vector_store.as_retriever(
                search_kwargs={"k": top_k}
            )
            docs = retriever.get_relevant_documents(query)
            logger.info(f"Retrieved {len(docs)} documents for query")
            return docs
        except Exception as e:
            logger.error(f"Failed to retrieve documents: {str(e)}")
            return []

    def _rerank(
            self,
            query: str,
            docs: List[Document],
            threshold: float,
    ) -> List[SearchHit]:
        if not self.cross_encoder:
            return [SearchHit(d) for d in docs]
        hits: List[SearchHit] = []
        failed = 0
        for doc in docs:
            try:
                score = self.cross_encoder.get_score(query, doc.page_content)
                if score >= threshold:
                    hits.append(SearchHit(doc, score))
            except Exception as e:
                logger.error(f"Failed to rerank document: {str(e)}")
                failed += 1
                hits.append(SearchHit(doc, 0.0))
        if failed > 0:
            logger.warning(f"Failed to rerank {failed}/{len(docs)} documents")
        hits.sort(key=lambda x: x.score or 0, reverse=True)
        return hits

    def search(self, params: SearchParameters) -> List[SearchHit]:
        try:
            docs = self._retrieve(params.query, params.top_k_retrieve)

            if not docs:
                logger.warning("No documents retrieved")
                return []

            if params.use_reranking:
                hits = self._rerank(
                    params.query,
                    docs,
                    params.rerank_threshold,
                )
                return hits[: params.top_k_reranking]
            return [SearchHit(d) for d in docs]
        except Exception as e:
            logger.error(f"Search failed: {str(e)}")
            return []

    def get_stats(self) -> Dict[str, Any]:
        try:
            collection = self._vector_store.get()
            return {
                "status": "ready",
                "num_documents": len(collection["ids"]),
                "persist_path": self.persist_path,
                "has_cross_encoder": self.cross_encoder is not None,
            }
        except Exception as e:
            logger.error(f"Failed to get stats: {str(e)}")
            return {
                "status": "error",
                "error": str(e),
                "num_documents": 0,
                "persist_path": self.persist_path,
                "has_cross_encoder": self.cross_encoder is not None,
            }