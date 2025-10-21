from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import json
import base64
from pathlib import Path
from openai import OpenAI
from app.core.config import OPENAI_API_KEY, LLM_PROVIDER
from app.core.gemini import gemini_service
from app.services.redis_service import redis_service
import logging
from PIL import Image
import io

logger = logging.getLogger(__name__)

class ImageError(Exception):
    """Custom exception for image processing errors"""
    pass

class ImageService:
    MAX_FILE_SIZE = 20 * 1024 * 1024  # 20MB
    ALLOWED_FORMATS = {'png', 'jpg', 'jpeg', 'gif', 'webp'}
    MAX_DIMENSION = 4096  # Max width or height
    
    def __init__(self):
        self.provider = LLM_PROVIDER
        self.openai_client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
        self.upload_dir = Path("uploads/images")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Image Service initialized with provider: {self.provider}")  # Already in English

    async def process_image_message(
        self, 
        message: str, 
        image_data: Optional[bytes] = None,
        image_filename: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        conversation_id = conversation_id or str(uuid.uuid4())
        
        try:
            # Load history
            history = await self.get_history(conversation_id)
            
            # Get current image path
            current_image_path = None
            current_image_data = None
            
            # If new image is uploaded, validate and save it
            if image_data and image_filename:
                current_image_path = await self._validate_and_save_image(
                    image_data, image_filename, conversation_id
                )
                current_image_data = image_data
            else:
                # Use the last image from history
                for msg in reversed(history):
                    if msg.get("image"):
                        current_image_path = msg["image"]
                        # Load image data from file
                        with open(current_image_path, "rb") as f:
                            current_image_data = f.read()
                        break
            
            if not current_image_path:
                return {
                    "response": "⚠️ **Error:** Please upload an image before asking questions.\n\n📸 **Supported formats:** PNG, JPG, JPEG, GIF, WebP\n📏 **Max size:** 20MB",  # Changed from Vietnamese
                    "conversation_id": conversation_id
                }
            
            # Check if image file exists
            if not Path(current_image_path).exists():
                return {
                    "response": "⚠️ **Error:** Image has been deleted or does not exist. Please upload again.",  # Changed from Vietnamese
                    "conversation_id": conversation_id
                }
            
            # Add user message
            user_msg = {
                "role": "user",
                "content": message,
                "image": current_image_path,
                "timestamp": datetime.now().isoformat()
            }
            history.append(user_msg)
            
            # Query with image based on provider
            try:
                if self.provider == "gemini":
                    response_text = await self._query_with_gemini(
                        message, 
                        current_image_data, 
                        history
                    )
                else:  # OpenAI
                    response_text = await self._query_with_openai(
                        message, 
                        current_image_path, 
                        history
                    )
                    
                if not response_text:
                    response_text = "⚠️ Sorry, I cannot analyze the image right now. Please try again later."  # Changed from Vietnamese
                    
            except Exception as e:
                logger.error(f"LLM API error: {e}")  # Already in English
                response_text = f"⚠️ **API Error:** Cannot analyze image. {str(e)}\n\n💡 Please try again later."  # Changed from Vietnamese
            
            # Add assistant message
            assistant_msg = {
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            }
            history.append(assistant_msg)
            
            # Save history
            await self.save_history(conversation_id, history)
            
            return {
                "response": response_text,
                "conversation_id": conversation_id,
                "image_url": f"/uploads/images/{Path(current_image_path).name}"
            }
            
        except ImageError as e:
            logger.error(f"Image processing error: {e}")  # Already in English
            return {
                "response": f"⚠️ **Image processing error:** {str(e)}",  # Changed from Vietnamese
                "conversation_id": conversation_id
            }
        except Exception as e:
            logger.error(f"Unexpected error in image processing: {e}")  # Already in English
            return {
                "response": "❌ **Unexpected error.** Please try again or upload a different image.",  # Changed from Vietnamese
                "conversation_id": conversation_id
            }

    async def _query_with_gemini(
        self, 
        message: str, 
        image_data: bytes, 
        history: List[Dict]
    ) -> str:
        """Query using Google Gemini"""
        # Build context from history
        context = "Conversation history:\n"  # Already in English
        for msg in history[-5:]:  # Last 5 messages
            if msg["role"] == "user":
                context += f"User: {msg['content']}\n"
            else:
                context += f"Assistant: {msg['content']}\n"
        
        full_prompt = f"{context}\nCurrent question: {message}\n\nPlease analyze the image and respond in English."  # Changed from Vietnamese
        
        return gemini_service.call_llm_with_image(
            prompt=full_prompt,
            image_data=image_data,
            temperature=0.7,
            max_tokens=500
        )

    async def _query_with_openai(
        self, 
        message: str, 
        image_path: str, 
        history: List[Dict]
    ) -> str:
        """Query using OpenAI GPT-4 Vision"""
        messages = self._build_openai_messages(history, image_path)
        
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
            max_tokens=500,
            timeout=30
        )
        return response.choices[0].message.content

    async def _validate_and_save_image(
        self, 
        image_data: bytes, 
        image_filename: str, 
        conversation_id: str
    ) -> str:
        """Validate and save uploaded image"""
        try:
            # Check file size
            if len(image_data) > self.MAX_FILE_SIZE:
                raise ImageError(f"Image too large (max {self.MAX_FILE_SIZE // (1024*1024)}MB). Please upload a smaller image.")  # Changed from Vietnamese
            
            # Check if empty
            if len(image_data) == 0:
                raise ImageError("Image file is empty. Please select a valid image.")  # Changed from Vietnamese
            
            # Check file extension
            file_ext = Path(image_filename).suffix.lower().lstrip('.')
            if file_ext not in self.ALLOWED_FORMATS:
                raise ImageError(f"Unsupported format: .{file_ext}\n\n✅ **Supported:** {', '.join(f'.{fmt}' for fmt in self.ALLOWED_FORMATS)}")  # Changed from Vietnamese
            
            # Validate image with PIL
            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()
                
                img = Image.open(io.BytesIO(image_data))
                width, height = img.size
                
                if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
                    raise ImageError(f"Image dimensions too large ({width}x{height}). Max: {self.MAX_DIMENSION}x{self.MAX_DIMENSION}px")  # Changed from Vietnamese
                
                if width == 0 or height == 0:
                    raise ImageError("Image has no content (0x0).")  # Changed from Vietnamese
                
                logger.info(f"Image validated: {image_filename}, {width}x{height}, format: {img.format}")  # Already in English
                
            except Exception as e:
                if isinstance(e, ImageError):
                    raise
                raise ImageError(f"File is not a valid image: {str(e)}")  # Changed from Vietnamese
            
            # Save image
            image_path = self.upload_dir / f"{conversation_id}_{image_filename}"
            with open(image_path, "wb") as f:
                f.write(image_data)
            
            return str(image_path)
            
        except ImageError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error validating image: {e}")  # Already in English
            raise ImageError(f"Unexpected error processing image: {str(e)}")  # Changed from Vietnamese

    def _build_openai_messages(self, history: List[Dict], image_path: str) -> List[Dict]:
        """Build messages for OpenAI API"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can analyze images. Respond in English."  # Already in English
                }
            ]
            
            for msg in history:
                if msg["role"] == "user":
                    messages.append({
                        "role": "user",
                        "content": [{"type": "text", "text": msg["content"]}]
                    })
                else:
                    messages.append({
                        "role": "assistant",
                        "content": msg["content"]
                    })
            
            # Add image to last user message
            with open(image_path, "rb") as img_file:
                base64_image = base64.b64encode(img_file.read()).decode('utf-8')
            
            image_format = Path(image_path).suffix[1:].lower()
            if image_format == 'jpg':
                image_format = 'jpeg'
            
            messages[-1]["content"].append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/{image_format};base64,{base64_image}"
                }
            })
            
            return messages
            
        except Exception as e:
            logger.error(f"Error building messages: {e}")  # Already in English
            raise ImageError(f"Error preparing image: {str(e)}")  # Changed from Vietnamese

    async def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        try:
            key = f"image_chat_history:{conversation_id}"
            data = await redis_service.get(key)
            return json.loads(data) if data else []
        except Exception as e:
            logger.error(f"Error getting history: {e}")  # Already in English
            return []

    async def save_history(self, conversation_id: str, history: List[Dict[str, Any]]):
        try:
            key = f"image_chat_history:{conversation_id}"
            await redis_service.set(key, json.dumps(history), expire=86400)
        except Exception as e:
            logger.error(f"Error saving history: {e}")  # Already in English

image_service = ImageService()