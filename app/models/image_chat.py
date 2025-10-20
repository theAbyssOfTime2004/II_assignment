from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from typing import Optional
from app.models.schemas import ImageChatResponse
from app.services.image_service import image_service

router = APIRouter()

@router.post("/send", response_model=ImageChatResponse)
async def send_image_message(
    message: str = Form(...),
    image: Optional[UploadFile] = File(None),
    conversation_id: Optional[str] = Form(None)
):
    try:
        image_data = None
        image_filename = None
        
        if image:
            image_data = await image.read()
            image_filename = image.filename
        
        response = await image_service.process_image_message(
            message=message,
            image_data=image_data,
            image_filename=image_filename,
            conversation_id=conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))