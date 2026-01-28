from typing import Dict

from app.graph.llm_client import LLMClient
from app.graph.prompts import get_prompt_template


class GenerateNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client
        self.prompt_template = get_prompt_template()

    def __call__(self, state: Dict) -> Dict:
        context_text = "\n".join([doc.page_content for doc in state.get("docs", [])])
        query = state["query"]
        prompt = self.prompt_template.format(context=context_text, query=query)
        state["answer"] = self.llm_client.generate(prompt)
        return state