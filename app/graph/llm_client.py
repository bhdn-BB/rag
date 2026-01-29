import os
import logging

from dotenv import load_dotenv
from google import genai
from google.genai import types

from app.models.parameters import LLMParams

logger = logging.getLogger("LLMClient")
logger.setLevel(logging.INFO)

load_dotenv()
class LLMClient:
    def __init__(self, params: LLMParams = LLMParams()):
        self.params = params
        self.api_key = os.getenv("LLM_API_KEY")
        if not self.api_key:
            raise ValueError("LLM_API_KEY не встановлено в середовищі")
        self.client = genai.Client(api_key=self.api_key)

    def generate(self, prompt: str) -> str:
        try:
            resp = self.client.models.generate_content(
                model=self.params.model_name,
                contents=prompt,
                config=types.GenerateContentConfig(
                    temperature=self.params.temperature,
                    max_output_tokens=self.params.max_output_tokens,
                ),
            )
            return resp.text.strip()

        except Exception as e:
            error_msg = str(e).lower()

            if "quota" in error_msg or "resource exhausted" in error_msg:
                logger.warning(f"Quota exceeded: {e}")
                return "⚠️ Ліміт генерації вичерпано."

            if "invalid" in error_msg and "api" in error_msg:
                logger.error(f"Invalid API key: {e}")
                return "⚠️ Помилка автентифікації."

            logger.error(f"LLM generation error: {e}")
            return "⚠️ Помилка генерації відповіді."