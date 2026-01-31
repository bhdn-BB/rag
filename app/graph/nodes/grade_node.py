import logging
from app.graph.llm_client import LLMClient
from app.graph.state_model import GraphState


logger = logging.getLogger("GradeNode")


class GradeNode:
    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def __call__(self, state: GraphState) -> dict:
        query = state.get("query", "")
        docs = state.get("docs", [])
        context_preview = " ".join([d.page_content[:] for d in docs]) if docs else "Немає документів"

        prompt = f"""
Тобі дано запит користувача та контекст документів. 
Визнач, чи достатньо наявних документів для відповіді на запит.

Запит: "{query}"

Контекст документів (попередній перегляд): "{context_preview}"

Відповідай лише одним словом:
- "ТАК" якщо можна відповісти на запит повністю за наявними документами
- "НІ" якщо інформації недостатньо
"""
        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            enough_data = "так" in response.lower() or "yes" in response.lower()
            logger.info(f"Оцінка документів: enough_data = {enough_data}")
            state["enough_data"] = enough_data
            return state
        except Exception as e:
            logger.error(f"Помилка оцінки документів: {str(e)}")
            state["enough_data"] = False
            return state
