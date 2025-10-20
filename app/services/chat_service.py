from typing import List, Dict, Any
from datetime import datetime
from app.core.llm import llm_service
from app.services.redis_service import redis_service
from app.models.schemas import ChatRequest, ChatResponse, Message
import uuid
import json

class ChatService:
    def __init__(self):
        self.system_prompt = "You are a helpful assistant. Respond in Vietnamese."

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        conversation_id = request.conversation_id or str(uuid.uuid4())
        
        # Load history
        history = await self.get_history(conversation_id)
        
        # Add user message
        user_msg = Message(role="user", content=request.message, timestamp=datetime.now().isoformat())
        history.append(user_msg.dict())
        
        # Build context
        context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-10:]])  # Last 10 messages
        
        # Call LLM
        response_text = llm_service.call_llm(self.system_prompt, f"History:\n{context}\nUser: {request.message}")
        if not response_text:
            response_text = "Xin lỗi, tôi không thể trả lời lúc này."
        
        # Add assistant message
        assistant_msg = Message(role="assistant", content=response_text, timestamp=datetime.now().isoformat())
        history.append(assistant_msg.dict())
        
        # Save history
        await self.save_history(conversation_id, history)
        
        return ChatResponse(response=response_text, conversation_id=conversation_id)

    async def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        key = f"chat_history:{conversation_id}"
        data = await redis_service.get(key)
        return json.loads(data) if data else []

    async def save_history(self, conversation_id: str, history: List[Dict[str, Any]]):
        key = f"chat_history:{conversation_id}"
        await redis_service.set(key, json.dumps(history), expire=86400)  # 1 day