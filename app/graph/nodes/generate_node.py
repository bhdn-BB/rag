from typing import List
from app.graph.llm_client import LLMClient
from app.graph.prompts import get_prompt_template
from langchain_core.documents import Document
from app.graph.state_model import GraphState, SourceInfo
import logging

logger = logging.getLogger("GenerateNode")


class GenerateNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_template = get_prompt_template()

    def _extract_source_info(self, doc: Document, idx: int, score: float = None) -> SourceInfo:
        metadata = doc.metadata or {}

        source = metadata.get("source") or metadata.get("title") or f"Document_{idx}"

        page = metadata.get("page") or metadata.get("page_number")
        if page is not None:
            try:
                page = int(page)
            except (ValueError, TypeError):
                page = None

        section = metadata.get("section") or metadata.get("header") or metadata.get("chapter")

        return SourceInfo(
            content=doc.page_content[:200] + "..." if len(doc.page_content) > 200 else doc.page_content,
            source=source,
            page=page,
            section=section,
            score=score,
            metadata=metadata
        )

    def _convert_to_document(self, item, idx: int) -> tuple[Document, float]:
        if hasattr(item, 'document') and hasattr(item, 'score'):
            return item.document, item.score

        elif isinstance(item, Document):
            score = item.metadata.get('score') if item.metadata else None
            return item, score
        else:
            logger.warning(f"Unknown item type at index {idx}: {type(item)}")
            return item, None

    def __call__(self, state: GraphState) -> dict:
        docs_raw = state["docs"]

        if not docs_raw:
            logger.warning("No documents provided for generation")
            return {
                "answer": "",
                "sources": []
            }

        context_parts = []
        sources_info: List[SourceInfo] = []

        for idx, item in enumerate(docs_raw, 1):
            doc, score = self._convert_to_document(item, idx)

            source_info = self._extract_source_info(doc, idx, score)
            sources_info.append(source_info)

            source_label = f"[{idx}] {source_info['source']}"
            if source_info['page'] is not None:
                source_label += f" (стор. {source_info['page']})"
            if source_info['section']:
                source_label += f" - {source_info['section']}"
            context_parts.append(f"{source_label}\n{doc.page_content}")
        context_text = "\n\n---\n\n".join(context_parts)
        prompt = self.prompt_template.format(
            context=context_text,
            query=state["query"]
        )
        try:
            answer = self.llm_client.generate(prompt)
            logger.info(f"Generated answer with {len(sources_info)} sources")
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            answer = ""

        return {
            "answer": answer,
            "sources": sources_info
        }