from typing import Dict, List
from app.graph.llm_client import LLMClient
from app.graph.prompts import get_prompt_template
from langchain_core.documents import Document
import logging

logger = logging.getLogger("GenerateNode")

class GenerateNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_template = get_prompt_template()

    def __call__(self, state: Dict) -> Dict:
        docs: List[Document] = state.get("docs", [])
        if not docs:
            state["answer"] = "На основі наданого контексту не можу відповісти на це питання."
            state["sources"] = []
            return state

        context_parts = []
        sources = []
        for idx, doc in enumerate(docs, 1):
            source = doc.metadata.get("source", f"Document_{idx}")
            sources.append(source)
            context_parts.append(f"[{source}] {doc.page_content}")

        context_text = "\n\n".join(context_parts)
        prompt = self.prompt_template.format(context=context_text, query=state["query"])

        try:
            answer = self.llm_client.generate(prompt)
        except Exception as e:
            logger.error(f"LLM generation error: {e}")
            answer = "Виникла помилка при генерації відповіді."

        state["answer"] = answer
        state["sources"] = sources
        return state
