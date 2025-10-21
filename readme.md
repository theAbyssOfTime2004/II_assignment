# Multi-Modal AI Chat Application

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.100.0-009688.svg)](https://fastapi.tiangolo.com/)
[![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o--mini-74AA9C.svg)](https://openai.com/)
[![LlamaIndex](https://img.shields.io/badge/LlamaIndex-Pandas-blueviolet)](https://www.llamaindex.ai/)
[![Redis](https://img.shields.io/badge/Redis-Cache-red)](https://redis.io/)

This is a multi-modal AI chat application built with a FastAPI backend. The system supports three distinct modes of interaction: standard text chat, image-based queries, and natural language analysis of CSV datasets.

---

## Core Functionality

The application provides a unified interface for three distinct AI-powered services.

### 1. Core Chat
- Standard, multi-turn text conversations.
- Conversation persistence is handled via Redis to maintain context across sessions.
- Simple request-response model for general-purpose Q&A.

### 2. Image Analysis
- Supports analysis of uploaded images (PNG, JPG, GIF, WebP).
- Users can submit queries related to the content of an uploaded image.
- Utilizes the GPT-4o-mini vision model for analysis.
- Includes server-side validation for file size, format, and dimensions.

### 3. CSV Data Analysis
- Supports data analysis from both local file uploads (`.csv`, `.txt`) and remote URLs.
- Translates natural language questions about the dataset into executable Pandas operations via LlamaIndex's `PandasQueryEngine`.
- Includes logic to automatically resolve GitHub "blob" URLs to their "raw" content equivalent.
- Provides detailed error handling for invalid URLs, parsing failures, oversized files, and incorrect content types.
- Capable of generating data summaries and statistics on demand.

### System and UI
- A single-page application built with vanilla HTML, CSS, and JavaScript.
- Features a toggle for dark and light color themes.
- Provides functionality to clear conversation history and reset the session.
- Implements an organized file storage system for uploaded and processed files.

## Technical Stack

- **Backend**: FastAPI, Uvicorn
- **AI / LLM**: OpenAI API (GPT-4o-mini), LlamaIndex
- **Data Handling**: Pandas
- **Database / Cache**: Redis
- **Frontend**: HTML, CSS, JavaScript
- **Supporting Libraries**: `python-dotenv`, `requests`, `Pillow`

## Project Structure

The project follows a modular structure to separate concerns.

```
.
├── app/
│   ├── core/         # Core components (LLM service, config)
│   ├── models/       # FastAPI routers and Pydantic schemas
│   ├── services/     # Business logic for each chat type
│   ├── utils/        # Helper functions
│   └── main.py       # FastAPI app entrypoint
├── templates/
│   └── chat.html     # Single-page frontend UI
├── uploads/          # Directory for user-uploaded files (in .gitignore)
│   ├── csv/
│   └── images/
├── .env              # Environment variables (local)
├── .gitignore        # Files to ignore in git
├── app.log           # Application log file (in .gitignore)
├── requirements.txt  # Python dependencies
└── readme.md         # This file
```

## Local Setup and Execution

### 1. Prerequisites

- Python 3.9+
- A running Redis instance (e.g., via Docker: `docker run -d -p 6379:6379 redis`)

### 2. Clone the Repository

```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 3. Environment Setup

It is recommended to use a virtual environment.

```bash
# For Linux/macOS
python3 -m venv venv
source venv/bin/activate

# For Windows
python -m venv venv
.\venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Configure Environment Variables

Create a `.env` file in the project root.

**Option 1: Using OpenAI**
```env
# OpenAI Configuration
OPENAI_API_KEY="sk-..."
OPENAI_MODEL_NAME="gpt-4o-mini"
LLM_PROVIDER="openai"

# Redis
REDIS_URL="redis://localhost:6379"
```

**Option 2: Using Google Gemini**
```env
# Google Gemini Configuration
GOOGLE_API_KEY="your_google_api_key"
GEMINI_MODEL_NAME="gemini-2.0-flash-exp"
LLM_PROVIDER="gemini"

# Redis
REDIS_URL="redis://localhost:6379"
```

**Note:** To get Google API key, visit https://ai.google.dev/

### 6. Run the Application

```bash
uvicorn app.main:app --reload
```

The application will be served at `http://127.0.0.1:8000`.

## Usage Guide

1.  Navigate to `http://127.0.0.1:8000` in a web browser.
2.  Select the desired interaction mode using the tabs: "Core Chat", "Image Chat", or "CSV Data Chat".
3.  For **Image Chat**, an image must be uploaded with or before the first query.
4.  For **CSV Chat**, either upload a file or provide a URL and use the "Process URL" button before submitting queries.
5.  The "Clear" button in the input area can be used to reset the current conversation.

## Quick Start with Docker 

Docker provides the fastest way to get the application running without manual dependency management.

### Prerequisites
- Docker and Docker Compose installed
- API keys (OpenAI or Google Gemini)

### Method 1: Automated Setup 
```bash
# 1. Clone repository
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name

# 2. Configure environment
cp .env.example .env
nano .env  # Add your API keys

# 3. Run automated script
chmod +x quick-start.sh
./quick-start.sh
```

**Access:** http://localhost:8000

### Method 2: Manual Docker Compose
```bash
# 1. Clone and setup
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
cp .env.example .env
nano .env  # Add API keys

# 2. Build and run
docker-compose up -d --build

# 3. View logs
docker-compose logs -f app
```

**Access:** http://localhost:8000

### Environment Configuration
Create `.env` from `.env.example`:

**OpenAI:**
```env
OPENAI_API_KEY=sk-your-key-here
OPENAI_MODEL_NAME=gpt-4o-mini
LLM_PROVIDER=openai
```

**Google Gemini:**
```env
GOOGLE_API_KEY=your-google-key-here
GEMINI_MODEL_NAME=gemini-2.0-flash-exp
LLM_PROVIDER=gemini
```

### Docker Commands
```bash
# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Restart
docker-compose restart

# Clean up
docker-compose down -v
```
## Demo Video

A short demo: 

<video width="560" height="315" controls>
  <source src="demo/demo.mp4" type="video/mp4">
</video>