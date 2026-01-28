import logging
from abc import abstractmethod
from typing import List, Literal, Protocol
import os
import re
from concurrent.futures import ThreadPoolExecutor

from langchain_core.documents import Document
from langchain_community.document_loaders import (
    WebBaseLoader,
    PyPDFLoader,
    Docx2txtLoader
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.models.parameters import ChunkingParameters, BatchWorker


logger = logging.getLogger("DBNParser")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s", "%Y-%m-%d %H:%M:%S")
ch.setFormatter(formatter)
logger.addHandler(ch)


class DocumentsParser(Protocol):
    @abstractmethod
    def load(
            self,
            source: str,
            source_type: Literal["url", "file"]
    ) -> List[Document]:
        ...


class DBNParser(DocumentsParser):

    def __init__(
            self,
            chunk_params: ChunkingParameters = ChunkingParameters(),
            processor_params: BatchWorker = BatchWorker(),
    ):
        self.chunk_params = chunk_params
        self.max_workers = processor_params.num_workers
        self.batch_size = processor_params.batch_size

        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_params.chunk_size,
            chunk_overlap=self.chunk_params.chunk_overlap,
            separators=self.chunk_params.h_separator,
        )

    def load(
            self,
            source: str,
            source_type: Literal["url", "file"],
    ) -> List[Document]:
        if source_type == "url":
            docs = self._load_from_url(source)
        elif source_type == "file":
            docs = self._load_from_file(source)
        else:
            raise ValueError("source_type must be 'url' or 'file'")
        return self._chunk_documents_parallel(docs)

    def _load_from_url(self, url: str) -> List[Document]:
        loader = WebBaseLoader(url)
        docs = loader.load()
        logger.info(f"Loaded {len(docs)} documents from URL: {url}")
        return docs

    def _load_from_file(self, path: str) -> List[Document]:
        ext = os.path.splitext(path)[1].lower()
        if ext == ".pdf":
            loader = PyPDFLoader(path)
        elif ext in [".docx", ".doc"]:
            loader = Docx2txtLoader(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}. Supported: .pdf, .docx, .doc")
        docs = loader.load()
        for doc in docs:
            doc.metadata.update({
                "source": os.path.basename(path),
                "file_type": ext,
            })
        if docs:
            logger.info(f"Loaded {len(docs)} documents from file: {os.path.basename(path)}")
            logger.info(f"First chunk preview: {docs[0].page_content[:100]}...")
        return docs

    def _process_single_doc(self, doc: Document) -> List[Document]:
        result = []
        for text_chunk in self.splitter.split_text(doc.page_content):
            if not text_chunk.strip():
                continue
            result.append(
                Document(
                    page_content=text_chunk,
                    metadata={
                        **doc.metadata,
                        "chunk_len": len(text_chunk),
                        "is_table": self._is_table(text_chunk),
                    },
                )
            )
        return result

    def _chunk_documents_parallel(self, docs: List[Document]) -> List[Document]:
        chunks = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            for i in range(0, len(docs), self.batch_size):
                batch = docs[i: i + self.batch_size]
                for doc_chunks in executor.map(self._process_single_doc, batch):
                    chunks.extend(doc_chunks)
        logger.info(f"Chunking complete. Total chunks: {len(chunks)}")
        if chunks:
            logger.info(f"First chunk preview: {chunks[0].page_content[:100]}...")
        return chunks

    @staticmethod
    def _is_table(text: str) -> bool:
        """Визначає, чи є текст таблицею (евристика)."""
        lines = text.splitlines()
        if len(lines) < 2:
            return False
        table_like = sum(1 for line in lines[:6]
                         if re.match(r"\s*[\|+\-].*[\|+\-]\s*", line))
        return table_like >= 2