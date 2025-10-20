from openai import OpenAI
from typing import Optional, List, Dict, Any
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL_NAME

class LLMService:
    def __init__(self):
        self.client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None

    def call_llm(self, system_prompt: str, user_message: str, temperature: float = 0.0, max_tokens: int = 500) -> Optional[str]:
        if not self.client:
            return None
        try:
            response = self.client.chat.completions.create(
                model=OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_message},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"LLM error: {e}")
            return None

llm_service = LLMService()