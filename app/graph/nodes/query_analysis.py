from app.graph.llm_client import LLMClient
from app.graph.state_model import GraphState
import logging

logger = logging.getLogger("QueryAnalysisNode")


class QueryAnalysisNode:

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def __call__(self, state: GraphState) -> dict:
        query = state["query"]

        prompt = f"""Проаналізуй наступний запит і визнач, чи потрібна для нього зовнішня інформація з документів, чи можна відповісти на основі загальних знань.

Запит: "{query}"

Відповідай ЛИШЕ одним словом:
- "ТАК" якщо запит потребує специфічної інформації з документів (факти, дані, конкретний контент)
- "НІ" якщо на запит можна відповісти загальними знаннями (математика, загальновідомі факти, привітання)

Приклади:
- "Що є в документі X?" → ТАК
- "Підсумуй звіт" → ТАК
- "Скільки буде 2+2?" → НІ
- "Привіт" → НІ
- "Хто є президентом України?" → НІ (загальні знання)
- "Які показники продажів у звіті за Q3?" → ТАК (потрібні документи)

Відповідь:"""
        try:
            response = self.llm_client.generate(prompt, max_tokens=10)
            need_external = "так" in response.lower() or "yes" in response.lower()
            logger.info(f"Аналіз запиту: need_external_info = {need_external}")
            return {
                "need_external_info": need_external
            }
        except Exception as e:
            logger.error(f"Помилка аналізу запиту: {str(e)}")
            return {
                "need_external_info": True
            }