from typing import List, Dict, Any
from datetime import datetime
from app.core.llm import llm_service
from app.services.redis_service import redis_service
from app.models.schemas import ChatRequest, ChatResponse, Message
import uuid
import json
import logging

logger = logging.getLogger(__name__)

class ChatService:
    MAX_MESSAGE_LENGTH = 4000
    MAX_HISTORY_LENGTH = 50
    
    def __init__(self):
        self.system_prompt = "You are a helpful assistant. Respond in English."  # Already in English
        logger.info("Chat service initialized")  # Already in English

    async def process_message(self, request: ChatRequest) -> ChatResponse:
        try:
            # Validate message
            if not request.message or not request.message.strip():
                return ChatResponse(
                    response="⚠️ Message is empty. Please enter content.",  # Changed from Vietnamese
                    conversation_id=request.conversation_id or str(uuid.uuid4())
                )
            
            if len(request.message) > self.MAX_MESSAGE_LENGTH:
                return ChatResponse(
                    response=f"⚠️ Message too long (max {self.MAX_MESSAGE_LENGTH} characters). Please shorten.",  # Changed from Vietnamese
                    conversation_id=request.conversation_id or str(uuid.uuid4())
                )
            
            conversation_id = request.conversation_id or str(uuid.uuid4())
            
            # Load history
            history = await self.get_history(conversation_id)
            
            # Limit history length
            if len(history) > self.MAX_HISTORY_LENGTH:
                logger.warning(f"History too long ({len(history)}), truncating")  # Already in English
                history = history[-self.MAX_HISTORY_LENGTH:]
            
            # Add user message
            user_msg = Message(role="user", content=request.message.strip(), timestamp=datetime.now().isoformat())
            history.append(user_msg.dict())
            
            # Build context
            context = "\n".join([f"{msg['role']}: {msg['content']}" for msg in history[-10:]])
            
            # Call LLM with error handling
            try:
                response_text = llm_service.call_llm(
                    self.system_prompt, 
                    f"History:\n{context}\nUser: {request.message}"
                )
                if not response_text:
                    response_text = "⚠️ Sorry, I cannot respond right now. Please try again later."  # Changed from Vietnamese
            except Exception as e:
                logger.error(f"LLM error: {e}")  # Already in English
                response_text = "❌ **AI connection error.** Please try again later."  # Changed from Vietnamese
            
            # Add assistant message
            assistant_msg = Message(role="assistant", content=response_text, timestamp=datetime.now().isoformat())
            history.append(assistant_msg.dict())
            
            # Save history
            await self.save_history(conversation_id, history)
            
            return ChatResponse(response=response_text, conversation_id=conversation_id)
            
        except Exception as e:
            logger.error(f"Unexpected error in chat service: {e}")  # Already in English
            return ChatResponse(
                response="❌ **Unexpected error.** Please try again.",  # Changed from Vietnamese
                conversation_id=request.conversation_id or str(uuid.uuid4())
            )

    async def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        try:
            key = f"chat_history:{conversation_id}"
            data = await redis_service.get(key)
            return json.loads(data) if data else []
        except Exception as e:
            logger.error(f"Error getting history: {e}")  # Already in English
            return []

    async def save_history(self, conversation_id: str, history: List[Dict[str, Any]]):
        try:
            key = f"chat_history:{conversation_id}"
            await redis_service.set(key, json.dumps(history), expire=86400)
        except Exception as e:
            logger.error(f"Error saving history: {e}")  # Already in English