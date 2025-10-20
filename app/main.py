from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from fastapi.staticfiles import StaticFiles
from app.models.chat import router as chat_router
from app.models.image_chat import router as image_chat_router
from app.models.csv_chat import router as csv_chat_router
from contextlib import asynccontextmanager
import os
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('app.log')
    ]
)

logger = logging.getLogger(__name__)

templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create upload directories
    os.makedirs("uploads/images", exist_ok=True)
    os.makedirs("uploads/csv", exist_ok=True)
    logger.info("Application startup: directories created")
    
    # Test Redis connection
    from app.services.redis_service import redis_service
    if await redis_service.ping():
        logger.info("Redis connection successful")
    else:
        logger.warning("Redis connection failed - app will work but without persistence")
    
    yield
    
    logger.info("Application shutdown")

app = FastAPI(
    title="Simple Chat AI",
    description="Core Chat with multi-turn history, Image Chat, and CSV Data Chat",
    version="2.0.0",
    lifespan=lifespan,
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount static files
app.mount("/uploads", StaticFiles(directory="uploads"), name="uploads")

# Include routers
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])
app.include_router(image_chat_router, prefix="/api/image-chat", tags=["Image Chat"])
app.include_router(csv_chat_router, prefix="/api/csv-chat", tags=["CSV Chat"])

# Health check
@app.get("/health")
async def health():
    from app.services.redis_service import redis_service
    redis_status = await redis_service.ping()
    return {
        "status": "ok",
        "redis": "connected" if redis_status else "disconnected"
    }

# Chat UI
from fastapi import Request
@app.get("/")
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})