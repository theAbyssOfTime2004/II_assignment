from typing import Optional, List, Dict, Any
from datetime import datetime
import uuid
import json
import base64
from pathlib import Path
from openai import OpenAI
from app.core.config import OPENAI_API_KEY
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
        self.client = OpenAI(api_key=OPENAI_API_KEY)
        self.upload_dir = Path("uploads/images")
        self.upload_dir.mkdir(parents=True, exist_ok=True)

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
            
            # If new image is uploaded, validate and save it
            if image_data and image_filename:
                current_image_path = await self._validate_and_save_image(
                    image_data, image_filename, conversation_id
                )
            else:
                # Use the last image from history
                for msg in reversed(history):
                    if msg.get("image"):
                        current_image_path = msg["image"]
                        break
            
            if not current_image_path:
                return {
                    "response": "⚠️ **Lỗi:** Vui lòng upload một ảnh trước khi đặt câu hỏi.\n\n📸 **Định dạng hỗ trợ:** PNG, JPG, JPEG, GIF, WebP\n📏 **Kích thước tối đa:** 20MB",
                    "conversation_id": conversation_id
                }
            
            # Check if image file exists
            if not Path(current_image_path).exists():
                return {
                    "response": "⚠️ **Lỗi:** Ảnh đã bị xóa hoặc không tồn tại. Vui lòng upload lại.",
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
            
            # Build conversation messages for OpenAI
            messages = self._build_messages(history, current_image_path)
            
            # Query with image
            try:
                response = self.client.chat.completions.create(
                    model="gpt-4o-mini",
                    messages=messages,
                    max_tokens=500,
                    timeout=30
                )
                response_text = response.choices[0].message.content
            except Exception as e:
                logger.error(f"OpenAI API error: {e}")
                response_text = f"⚠️ **Lỗi API:** Không thể phân tích ảnh. {str(e)}\n\n💡 Vui lòng thử lại sau."
            
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
            logger.error(f"Image processing error: {e}")
            return {
                "response": f"⚠️ **Lỗi xử lý ảnh:** {str(e)}",
                "conversation_id": conversation_id
            }
        except Exception as e:
            logger.error(f"Unexpected error in image processing: {e}")
            return {
                "response": "❌ **Lỗi không xác định.** Vui lòng thử lại hoặc upload ảnh khác.",
                "conversation_id": conversation_id
            }

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
                raise ImageError(f"Ảnh quá lớn (max {self.MAX_FILE_SIZE // (1024*1024)}MB). Vui lòng upload ảnh nhỏ hơn.")
            
            # Check if empty
            if len(image_data) == 0:
                raise ImageError("File ảnh trống. Vui lòng chọn ảnh hợp lệ.")
            
            # Check file extension
            file_ext = Path(image_filename).suffix.lower().lstrip('.')
            if file_ext not in self.ALLOWED_FORMATS:
                raise ImageError(f"Định dạng không hỗ trợ: .{file_ext}\n\n✅ **Hỗ trợ:** {', '.join(f'.{fmt}' for fmt in self.ALLOWED_FORMATS)}")
            
            # Validate image with PIL
            try:
                img = Image.open(io.BytesIO(image_data))
                img.verify()  # Verify it's a valid image
                
                # Re-open for further processing (verify() closes the file)
                img = Image.open(io.BytesIO(image_data))
                
                # Check dimensions
                width, height = img.size
                if width > self.MAX_DIMENSION or height > self.MAX_DIMENSION:
                    raise ImageError(f"Kích thước ảnh quá lớn ({width}x{height}). Max: {self.MAX_DIMENSION}x{self.MAX_DIMENSION}px")
                
                # Check if image has content
                if width == 0 or height == 0:
                    raise ImageError("Ảnh không có nội dung (0x0).")
                
                logger.info(f"Image validated: {image_filename}, {width}x{height}, format: {img.format}")
                
            except Exception as e:
                if isinstance(e, ImageError):
                    raise
                raise ImageError(f"File không phải là ảnh hợp lệ: {str(e)}")
            
            # Save image
            image_path = self.upload_dir / f"{conversation_id}_{image_filename}"
            with open(image_path, "wb") as f:
                f.write(image_data)
            
            return str(image_path)
            
        except ImageError:
            raise
        except Exception as e:
            logger.error(f"Unexpected error validating image: {e}")
            raise ImageError(f"Lỗi không xác định khi xử lý ảnh: {str(e)}")

    def _build_messages(self, history: List[Dict], image_path: str) -> List[Dict]:
        """Build messages for OpenAI API"""
        try:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant that can analyze images. Respond in Vietnamese."
                }
            ]
            
            # Add conversation history
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
            logger.error(f"Error building messages: {e}")
            raise ImageError(f"Lỗi khi chuẩn bị ảnh: {str(e)}")

    async def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        try:
            key = f"image_chat_history:{conversation_id}"
            data = await redis_service.get(key)
            return json.loads(data) if data else []
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    async def save_history(self, conversation_id: str, history: List[Dict[str, Any]]):
        try:
            key = f"image_chat_history:{conversation_id}"
            await redis_service.set(key, json.dumps(history), expire=86400)
        except Exception as e:
            logger.error(f"Error saving history: {e}")

image_service = ImageService()