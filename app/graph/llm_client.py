import os
import logging

from google import genai
from google.genai import types

from app.models.parameters import LLMParams

logger = logging.getLogger("LLMClient")
logger.setLevel(logging.INFO)


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
            logger.error(f"LLM generation error: {e}")
