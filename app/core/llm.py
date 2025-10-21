from openai import OpenAI
from typing import Optional, List, Dict, Any
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL_NAME, LLM_PROVIDER
from app.core.gemini import gemini_service
import logging

logger = logging.getLogger(__name__)

class LLMService:
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.system_prompt = "You are a helpful assistant. Respond in English."  # Changed from Vietnamese
        logger.info(f"LLM Service initialized with provider: {self.provider}")

    def call_llm(
        self, 
        system_prompt: str, 
        user_message: str, 
        temperature: float = 1.0, 
        max_tokens: int = 500
    ) -> Optional[str]:
        """Call LLM based on configured provider"""
        
        if self.provider == "gemini":
            return gemini_service.call_llm(
                system_prompt, 
                user_message, 
                temperature, 
                max_tokens
            )
        
        elif self.provider == "openai":
            if not self.openai_client:
                logger.error("OpenAI client not initialized")
                return None
            
            try:
                response = self.openai_client.chat.completions.create(
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
                logger.error(f"OpenAI error: {e}")
                return None
        
        else:
            logger.error(f"Unknown LLM provider: {self.provider}")
            return None

llm_service = LLMService()