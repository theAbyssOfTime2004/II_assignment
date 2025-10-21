import google.generativeai as genai
from typing import Optional, List, Dict, Any
from app.core.config import GOOGLE_API_KEY, GEMINI_MODEL_NAME
import logging

logger = logging.getLogger(__name__)

class GeminiService:
    def __init__(self):
        if GOOGLE_API_KEY:
            genai.configure(api_key=GOOGLE_API_KEY)
            self.model = genai.GenerativeModel(GEMINI_MODEL_NAME)
            logger.info(f"Gemini initialized with model: {GEMINI_MODEL_NAME}")
        else:
            self.model = None
            logger.warning("Gemini API key not found")

    def call_llm(
        self, 
        system_prompt: str, 
        user_message: str, 
        temperature: float = 1.0, 
        max_tokens: int = 500
    ) -> Optional[str]:
        if not self.model:
            return None
        
        try:
            # Combine system prompt and user message for Gemini
            full_prompt = f"{system_prompt}\n\nUser: {user_message}"
            
            # Configure generation
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = self.model.generate_content(
                full_prompt,
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini error: {e}")
            return None

    def call_llm_with_image(
        self,
        prompt: str,
        image_data: bytes,
        temperature: float = 1.0,
        max_tokens: int = 500
    ) -> Optional[str]:
        """Call Gemini with image input"""
        if not self.model:
            return None
        
        try:
            from PIL import Image
            import io
            
            # Convert bytes to PIL Image
            image = Image.open(io.BytesIO(image_data))
            
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            
            response = self.model.generate_content(
                [prompt, image],
                generation_config=generation_config
            )
            
            return response.text
            
        except Exception as e:
            logger.error(f"Gemini image error: {e}")
            return None

gemini_service = GeminiService()