from pydantic import BaseModel
from typing import Optional, List

class ChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: str = "default_user"

class ChatResponse(BaseModel):
    response: str
    conversation_id: str

class Message(BaseModel):
    role: str  # "user" or "assistant"
    content: str
    timestamp: str

class ImageChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    user_id: str = "default_user"

class ImageChatResponse(BaseModel):
    response: str
    conversation_id: str
    image_url: Optional[str] = None

class CSVChatRequest(BaseModel):
    message: str
    conversation_id: Optional[str] = None
    csv_url: Optional[str] = None
    user_id: str = "default_user"

class CSVChatResponse(BaseModel):
    response: str
    conversation_id: str
    chart_data: Optional[dict] = None