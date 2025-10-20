from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from app.models.schemas import CSVChatRequest, CSVChatResponse
from app.services.csv_service import csv_service
import uuid

router = APIRouter()

@router.post("/send", response_model=CSVChatResponse)
async def send_csv_message(
    message: str = Form(...),
    csv_file: UploadFile = File(None),
    csv_url: str = Form(None),
    conversation_id: str = Form(None)
):
    try:
        # **NEW: Validate input**
        if csv_file and csv_url:
            return CSVChatResponse(
                response="⚠️ **Lỗi đầu vào:** Vui lòng chỉ chọn một trong hai: upload file HOẶC cung cấp URL.",
                conversation_id=conversation_id or str(uuid.uuid4())
            )
        
        csv_data = await csv_file.read() if csv_file else None
        response = await csv_service.process_csv_message(
            message=message,
            csv_data=csv_data,
            csv_url=csv_url,
            conversation_id=conversation_id
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))