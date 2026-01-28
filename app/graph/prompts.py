from langchain_core.prompts import PromptTemplate

SYSTEM_PROMPT = """Ти — висококваліфікований асистент для відповідей на питання користувача.
Твоя задача — відповідати **тільки на основі наданого контексту**.
Не вигадуй нічого, не додавай власних думок.
Якщо інформації недостатньо — скажи: "На основі наданого контексту не можу відповісти на це питання."
"""

USER_PROMPT_TEMPLATE = """Контекст документів:
{context}

Питання користувача:
{query}

Дай коротку, точну відповідь лише на основі контексту.
"""

def get_prompt_template() -> PromptTemplate:
    return PromptTemplate(
        input_variables=["context", "query"],
        template=SYSTEM_PROMPT + "\n\n" + USER_PROMPT_TEMPLATE,
    )
