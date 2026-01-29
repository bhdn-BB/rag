import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings

from app.models.parameters import SearchHit, SearchParameters
from app.services.embedders import HFBiEmbedder, HFCrossEncoder

logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: logging.Logger = logging.getLogger(__name__)


class HFEmbeddingWrapper(Embeddings):
    def __init__(
        self,
        embedder: HFBiEmbedder,
        batch_size: int = 16,
    ) -> None:
        self.embedder: HFBiEmbedder = embedder
        self.batch_size: int = batch_size

    def embed_documents(
        self,
        texts: List[str],
    ) -> List[List[float]]:
        embeddings: List[List[float]] = []

        for start in range(0, len(texts), self.batch_size):
            batch: List[str] = texts[start : start + self.batch_size]

            batch_embeddings: List[List[float]] = [
                self.embedder.get_embedding(text).cpu().tolist()
                for text in batch
            ]

            embeddings.extend(batch_embeddings)

            logger.info(
                "Embedded batch | index=%d size=%d",
                start // self.batch_size + 1,
                len(batch),
            )

        return embeddings

    def embed_query(self, text: str) -> List[float]:
        embedding: List[float] = (
            self.embedder.get_embedding(text).cpu().tolist()
        )

        logger.info("Embedded query | preview=%s", text[:50])
        return embedding


class VectorMemory:
    def __init__(
        self,
        bi_embedder: HFBiEmbedder,
        cross_encoder: Optional[HFCrossEncoder] = None,
        persist_path: str = "./data/chroma_index",
        collection_name: str = "documents",
    ):
        self.persist_path = persist_path
        self.cross_encoder = cross_encoder
        Path(self.persist_path).mkdir(
            parents=True,
            exist_ok=True,
        )

        embedding_wrapper: HFEmbeddingWrapper = HFEmbeddingWrapper(
            bi_embedder,
        )

        self._vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=self.persist_path,
            embedding_function=embedding_wrapper,
        )
        logger.info(
            "VectorMemory initialized | path=%s collection=%s",
            self.persist_path,
            collection_name,
        )

    def add_documents(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents to add")
            return
        self._vector_store.add_documents(documents)
        logger.info(
            "Documents added | count=%d",
            len(documents),
        )

    def delete_documents(
        self,
        filter_metadata: Dict[str, Any],
    ) -> None:
        self._vector_store.delete(where=filter_metadata)
        logger.info(
            "Documents deleted | filter=%s",
            filter_metadata,
        )
    def clear(self) -> None:
        collection = self._vector_store._collection
        ids: List[str] = collection.get().get("ids", [])

        if not ids:
            logger.warning("Vector store already empty")
            return

        collection.delete(ids=ids)

        logger.warning(
            "Vector store cleared | deleted=%d",
            len(ids),
        )

    def _retrieve(
        self,
        query: str,
        top_k: int,
    ) -> List[Document]:
        retriever = self._vector_store.as_retriever(
            search_kwargs={"k": top_k},
        )
        return retriever.invoke(query)

    def _rerank(
        self,
        query: str,
        documents: List[Document],
        threshold: float,
    ) -> List[SearchHit]:
        if not self.cross_encoder:
            return [SearchHit(doc) for doc in documents]

        hits: List[SearchHit] = []

        for doc in documents:
            score: float = self.cross_encoder.get_score(
                query,
                doc.page_content,
            )

            if score >= threshold:
                hits.append(SearchHit(doc, score))

        hits.sort(
            key=lambda hit: hit.score or 0.0,
            reverse=True,
        )

        return hits

    def search(
        self,
        params: SearchParameters,
    ) -> List[SearchHit]:

        documents: List[Document] = self._retrieve(
            params.query,
            params.top_k_retrieve,
        )

        if not params.use_reranking:
            return [SearchHit(doc) for doc in documents]

        hits: List[SearchHit] = self._rerank(
            params.query,
            documents,
            params.rerank_threshold,
        )

        return hits[: params.top_k_reranking]

    def get_stats(self) -> Dict[str, Any]:
        collection = self._vector_store._collection
        ids: List[str] = collection.get().get("ids", [])
        return {
            "status": "ready",
            "num_documents": len(ids),
            "persist_path": self.persist_path,
            "has_cross_encoder": self.cross_encoder is not None,
        }