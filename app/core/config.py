import os
from dotenv import load_dotenv

load_dotenv()

# OpenAI Config
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")

# Google Gemini Config
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-2.0-flash-exp")

# LLM Provider Selection
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "openai")  # "openai" or "gemini"

# Redis Config
REDIS_URL = os.getenv("REDIS_URL", "redis://localhost:6379")