from app.graph.llm_client import LLMClient
from app.graph.state_model import GraphState
import logging

logger = logging.getLogger("RewriteQueryNode")


class RewriteQueryNode:

    def __init__(self, llm_client: LLMClient):
        self.llm_client = llm_client

    def __call__(self, state: GraphState) -> dict:
        original_query = state["input_query"]
        current_query = state["query"]

        logger.info(f"Rewriting query (attempt {state['rewrite_attempts'] + 1})")

        prompt = f"""Перефразуй наступний запит, щоб зробити його більш ефективним для семантичного пошуку в базі документів.

Оригінальний запит: "{original_query}"
Поточний запит: "{current_query}"

Зроби запит більш конкретним, додай релевантні ключові слова та розшифруй скорочення якщо потрібно.
Відповідай ЛИШЕ переформульованим запитом, нічого іншого.

Переформульований запит:"""
        try:
            rewritten_query = self.llm_client.generate(prompt).strip()
            if not rewritten_query:
                rewritten_query = original_query
            logger.info(f"Query rewritten: '{current_query}' -> '{rewritten_query}'")
            return {
                "query": rewritten_query,
                "rewrite_attempts": 1
            }
        except Exception as e:
            logger.error(f"Query rewrite failed: {str(e)}")
            return {
                "query": f"{current_query} пояснення визначення опис",
                "rewrite_attempts": 1
            }