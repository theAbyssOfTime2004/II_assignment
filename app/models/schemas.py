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