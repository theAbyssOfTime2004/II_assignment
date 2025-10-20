from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.templating import Jinja2Templates
from app.models.chat import router as chat_router
from contextlib import asynccontextmanager
import os

templates = Jinja2Templates(directory="templates")

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Ultra-fast startup - no heavy operations
    yield

app = FastAPI(
    title="Simple Chat AI",
    description="Core Chat with multi-turn history and streaming",
    version="1.0.0",
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

# Include chat router
app.include_router(chat_router, prefix="/api/chat", tags=["Chat"])

# Health check
@app.get("/health")
async def health():
    return {"status": "ok"}

# Chat UI
from fastapi import Request
@app.get("/")
async def chat_ui(request: Request):
    return templates.TemplateResponse("chat.html", {"request": request})