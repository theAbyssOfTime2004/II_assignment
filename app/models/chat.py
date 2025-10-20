from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from app.models.schemas import ChatRequest, ChatResponse
from app.services.chat_service import ChatService
import json

router = APIRouter()
chat_service = ChatService()

@router.post("/send", response_model=ChatResponse)
async def send_message(request: ChatRequest):
    try:
        response = await chat_service.process_message(request)
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/stream/{conversation_id}")
async def stream_chat(conversation_id: str):
    async def event_generator():
        # Simulate streaming (kế thừa logic từ agent.py)
        history = await chat_service.get_history(conversation_id)
        for msg in history:
            yield f"data: {json.dumps(msg)}\n\n"
    return StreamingResponse(event_generator(), media_type="text/event-stream")