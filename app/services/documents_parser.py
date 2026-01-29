import io
import logging
import os
import re
from abc import abstractmethod
from concurrent.futures import ThreadPoolExecutor
from typing import Iterable, List, Literal, Protocol

import requests
from langchain_community.document_loaders import (
    Docx2txtLoader,
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.parameters import BatchWorker, ChunkingParameters


logging.basicConfig(
    level=logging.INFO,
    format="[%(levelname)s] %(asctime)s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

logger: logging.Logger = logging.getLogger(__name__)


class DocumentsParser(Protocol):
    @abstractmethod
    def load(
        self,
        source: str,
        source_type: Literal["url", "file"],
    ) -> List[Document]:
        ...


class DBNParser(DocumentsParser):
    def __init__(
        self,
        chunk_params: ChunkingParameters = ChunkingParameters(),
        processor_params: BatchWorker = BatchWorker(),
    ) -> None:
        self.chunk_params: ChunkingParameters = chunk_params
        self.max_workers: int = processor_params.num_workers
        self.batch_size: int = processor_params.batch_size

        self.splitter: RecursiveCharacterTextSplitter = (
            RecursiveCharacterTextSplitter(
                chunk_size=self.chunk_params.chunk_size,
                chunk_overlap=self.chunk_params.chunk_overlap,
                separators=self.chunk_params.h_separator,
            )
        )

        logger.info(
            "DBNParser initialized | chunk_size=%d overlap=%d workers=%d batch=%d",
            self.chunk_params.chunk_size,
            self.chunk_params.chunk_overlap,
            self.max_workers,
            self.batch_size,
        )

    def load(
        self,
        source: str,
        source_type: Literal["url", "file"],
    ) -> List[Document]:

        logger.info(
            "Loading source | type=%s source=%s",
            source_type,
            source,
        )

        if source_type == "url":
            if source.lower().endswith(".pdf"):
                return self._load_pdf_from_url(source)

            documents: List[Document] = self._load_from_url(source)

        elif source_type == "file":
            documents = self._load_from_file(source)

        else:
            raise ValueError("source_type must be 'url' or 'file'")

        return self._chunk_documents_parallel(documents)

    def _load_pdf_from_url(self, url: str) -> List[Document]:
        logger.info("Downloading PDF from URL: %s", url)
        try:
            response: requests.Response = requests.get(url, timeout=60)
            response.raise_for_status()
        except Exception:
            logger.exception("Failed to download PDF from URL: %s", url)
            return []
        try:
            pdf_file: io.BytesIO = io.BytesIO(response.content)
            loader: PyPDFLoader = PyPDFLoader(pdf_file)
            documents: List[Document] = loader.load()

            for doc in documents:
                doc.metadata.update(
                    {
                        "source": url,
                        "file_type": ".pdf",
                    }
                )

            logger.info(
                "Loaded PDF | pages=%d url=%s",
                len(documents),
                url,
            )

            return self._chunk_documents_parallel(documents)

        except Exception:
            logger.exception("Failed to parse PDF from URL: %s", url)
            return []

    def _load_from_url(self, url: str) -> List[Document]:
        loader: WebBaseLoader = WebBaseLoader(url)
        documents: List[Document] = loader.load()

        logger.info(
            "Loaded documents from URL | count=%d url=%s",
            len(documents),
            url,
        )

        return documents

    def _load_from_file(self, path: str) -> List[Document]:
        extension: str = os.path.splitext(path)[1].lower()

        loader: PyPDFLoader | Docx2txtLoader

        if extension == ".pdf":
            loader = PyPDFLoader(path)
        elif extension in {".docx", ".doc"}:
            loader = Docx2txtLoader(path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

        documents: List[Document] = loader.load()

        for doc in documents:
            doc.metadata.update(
                {
                    "source": os.path.basename(path),
                    "file_type": extension,
                }
            )

        logger.info(
            "Loaded documents from file | file=%s pages=%d",
            os.path.basename(path),
            len(documents),
        )

        return documents

    def _process_single_document(
        self,
        document: Document,
    ) -> List[Document]:
        chunks: List[Document] = []

        for text_chunk in self.splitter.split_text(
            document.page_content,
        ):
            if not text_chunk.strip():
                continue

            chunks.append(
                Document(
                    page_content=text_chunk,
                    metadata={
                        **document.metadata,
                        "chunk_len": len(text_chunk),
                        "is_table": self._is_table(text_chunk),
                    },
                )
            )

        return chunks

    def _chunk_documents_parallel(
        self,
        documents: List[Document],
    ) -> List[Document]:

        logger.info(
            "Chunking started | docs=%d workers=%d batch_size=%d",
            len(documents),
            self.max_workers,
            self.batch_size,
        )

        chunks: List[Document] = []

        with ThreadPoolExecutor(
            max_workers=self.max_workers,
        ) as executor:
            for start in range(0, len(documents), self.batch_size):
                batch: List[Document] = documents[
                    start : start + self.batch_size
                ]

                for doc_chunks in executor.map(
                    self._process_single_document,
                    batch,
                ):
                    chunks.extend(doc_chunks)

        logger.info(
            "Chunking complete | total_chunks=%d",
            len(chunks),
        )

        return chunks

    @staticmethod
    def _is_table(text: str) -> bool:
        lines: List[str] = text.splitlines()

        if len(lines) < 2:
            return False

        table_like_lines: int = sum(
            1
            for line in lines[:6]
            if re.match(r"\s*[\|+\-].*[\|+\-]\s*", line)
        )
        return table_like_lines >= 2
