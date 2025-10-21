from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
from pathlib import Path
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings
from io import StringIO
import uuid
import json
import pandas as pd
import numpy as np
import requests
import logging
import os

from app.services.redis_service import redis_service
from app.core.llm import llm_service

logger = logging.getLogger(__name__)

class CSVError(Exception):
    """Custom exception for CSV processing errors"""
    pass

class CSVService:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_ROWS = 1_000_000
    ALLOWED_EXTENSIONS = ['.csv', '.txt']
    
    def __init__(self):
        self.upload_dir = Path("uploads/csv")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure llama-index with OpenAI model
        try:
            OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
            OPENAI_MODEL_NAME = os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
            Settings.llm = LlamaOpenAI(
                api_key=OPENAI_API_KEY,
                model=OPENAI_MODEL_NAME,
                temperature=0.7
            )
            logger.info(f"LlamaIndex configured to use {OPENAI_MODEL_NAME}")
        except Exception as e:
            logger.error(f"Failed to configure LlamaIndex: {e}")

    async def process_csv_message(
        self,
        message: str,
        csv_data: Optional[bytes] = None,
        csv_url: Optional[str] = None,
        conversation_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process CSV chat message with unified RAG + LLM approach:
        - Both URL and Upload use text extraction + LLM analysis
        - No pattern matching, pure conversational AI
        """
        conversation_id = conversation_id or str(uuid.uuid4())
        
        try:
            # Check for conflicting inputs
            if csv_data and csv_url:
                return {
                    "response": (
                        "⚠️ **Error:** Cannot process both uploaded file and URL simultaneously.\n\n"
                        "💡 **Hint:** Please choose only one:\n"
                        "- Upload a CSV file directly, OR\n"
                        "- Provide a CSV URL"
                    ),
                    "conversation_id": conversation_id
                }
            
            # Load chat history
            history = await self.get_history(conversation_id)
            
            # Add user message
            user_msg = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            history.append(user_msg)
            
            # **UNIFIED MODE: RAG + LLM for both URL and Upload**
            if csv_url:
                response_text, chart_data = await self._process_csv_url_via_rag(
                    message, csv_url, conversation_id, history
                )
            elif csv_data:
                response_text, chart_data = await self._process_csv_upload(
                    message, csv_data, conversation_id, history
                )
            else:
                # Continue conversation
                response_text, chart_data = await self._continue_conversation(
                    message, conversation_id, history
                )
            
            # Add assistant message
            assistant_msg = {
                "role": "assistant",
                "content": response_text,
                "timestamp": datetime.now().isoformat()
            }
            history.append(assistant_msg)
            
            # Save history
            await self.save_history(conversation_id, history)
            
            return {
                "response": response_text,
                "conversation_id": conversation_id,
                "chart_data": chart_data
            }
            
        except CSVError as e:
            logger.error(f"CSV processing error: {e}")
            return {
                "response": f"⚠️ **CSV Error:** {str(e)}",
                "conversation_id": conversation_id
            }
        except Exception as e:
            logger.error(f"Unexpected error in CSV processing: {e}")
            return {
                "response": "❌ **Unexpected error.** Please try again or upload another CSV file.",
                "conversation_id": conversation_id
            }

    async def _process_csv_url_via_rag(
        self,
        message: str,
        csv_url: str,
        conversation_id: str,
        history: List[Dict]
    ) -> Tuple[str, Optional[dict]]:
        """Process CSV URL using RAG + LLM (same as upload mode)"""
        logger.info(f"Processing CSV URL via RAG: {csv_url}")
        
        try:
            # Step 1: Download CSV from URL
            csv_path = self.upload_dir / f"{conversation_id}.csv"
            df = await self._load_from_url(csv_url, csv_path)
            
            if df is None:
                return (
                    "⚠️ **Error:** Cannot load CSV from URL. Please check the link.",
                    None
                )
            
            # Step 2: Extract CSV content as text
            csv_text = await self._extract_csv_text(csv_path)
            
            # Step 3: Store in Redis
            await self._store_csv_context(conversation_id, csv_text, csv_path)
            
            # Step 4: Build prompt with CSV context
            prompt = await self._build_rag_prompt(message, conversation_id, history)
            
            # Step 5: Call LLM
            response_text = llm_service.call_llm(
                system_prompt="You are a data analysis assistant. Analyze the provided CSV data and answer user questions.",
                user_message=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            if not response_text:
                response_text = "⚠️ Sorry, I cannot analyze the CSV right now. Please try again later."
            
            return response_text, None
            
        except Exception as e:
            logger.error(f"Error processing CSV URL: {e}")
            return f"⚠️ **Error:** {str(e)}", None

    async def _process_csv_upload(
        self,
        message: str,
        csv_data: bytes,
        conversation_id: str,
        history: List[Dict]
    ) -> Tuple[str, Optional[dict]]:
        """Mode 2: Text extraction + RAG for uploaded CSV files"""
        logger.info(f"Processing uploaded CSV via text extraction")
        
        try:
            # Step 1: Validate and save CSV
            csv_path = self.upload_dir / f"{conversation_id}.csv"
            await self._validate_and_save_csv(csv_data, csv_path)
            
            # Step 2: Extract CSV content as text
            csv_text = await self._extract_csv_text(csv_path)
            
            # Step 3: Store in Redis for this conversation (simulating vector DB)
            await self._store_csv_context(conversation_id, csv_text, csv_path)
            
            # Step 4: Build prompt with CSV context
            prompt = await self._build_rag_prompt(message, conversation_id, history)
            
            # Step 5: Call LLM
            response_text = llm_service.call_llm(
                system_prompt="You are a data analysis assistant. Analyze the provided CSV data and answer user questions.",
                user_message=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            if not response_text:
                response_text = "⚠️ Sorry, I cannot analyze the CSV right now. Please try again later."
            
            return response_text, None
            
        except Exception as e:
            logger.error(f"Error in CSV upload processing: {e}")
            return f"⚠️ **Error processing uploaded CSV:** {str(e)}", None

    async def _continue_conversation(
        self,
        message: str,
        conversation_id: str,
        history: List[Dict]
    ) -> Tuple[str, Optional[dict]]:
        """Mode 3: Continue conversation using stored CSV context"""
        logger.info(f"Continuing CSV conversation for {conversation_id}")
        
        try:
            # Check if CSV context exists
            csv_context = await self._get_csv_context(conversation_id)
            
            if not csv_context:
                return (
                    "⚠️ **Error:** Please upload a CSV file or provide a valid URL first.\n\n"
                    "📄 **Supported formats:** .csv, .txt\n"
                    "📏 **Max size:** 50MB",
                    None
                )
            
            # Build prompt with existing context
            prompt = await self._build_rag_prompt(message, conversation_id, history)
            
            # Call LLM
            response_text = llm_service.call_llm(
                system_prompt="You are a data analysis assistant. Continue the conversation about the CSV data.",
                user_message=prompt,
                temperature=0.7,
                max_tokens=1000
            )
            
            if not response_text:
                response_text = "⚠️ Sorry, I cannot respond right now. Please try again later."
            
            return response_text, None
            
        except Exception as e:
            logger.error(f"Error continuing conversation: {e}")
            return f"⚠️ **Error:** {str(e)}", None

    async def _validate_and_save_csv(self, csv_data: bytes, csv_path: Path):
        """Validate and save uploaded CSV"""
        # Validate size
        if len(csv_data) > self.MAX_FILE_SIZE:
            raise CSVError(f"File too large (max {self.MAX_FILE_SIZE // (1024*1024)}MB)")
        
        if len(csv_data) == 0:
            raise CSVError("Empty CSV file")
        
        # Save to disk
        with open(csv_path, "wb") as f:
            f.write(csv_data)
        
        logger.info(f"CSV saved to {csv_path}")

    async def _extract_csv_text(self, csv_path: Path) -> str:
        """Extract CSV content as plain text (simulating LlamaIndexFileProcessor)"""
        try:
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Build text representation
            text_parts = []
            
            # Add basic info
            text_parts.append(f"CSV File: {csv_path.name}")
            text_parts.append(f"Total Rows: {len(df)}")
            text_parts.append(f"Total Columns: {len(df.columns)}")
            text_parts.append(f"Columns: {', '.join(df.columns.tolist())}")
            text_parts.append("\n--- Data Preview ---")
            
            # Add first 20 rows as text
            text_parts.append(df.head(20).to_string())
            
            # Add column info
            text_parts.append("\n--- Column Information ---")
            for col in df.columns:
                dtype = df[col].dtype
                non_null = df[col].count()
                missing = df[col].isnull().sum()
                text_parts.append(f"- {col}: type={dtype}, non-null={non_null}, missing={missing}")
            
            # Add numeric stats if available
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            if len(numeric_cols) > 0:
                text_parts.append("\n--- Numeric Column Statistics ---")
                text_parts.append(df[numeric_cols].describe().to_string())
            
            csv_text = "\n".join(text_parts)
            logger.info(f"Extracted {len(csv_text)} characters from CSV")
            
            return csv_text
            
        except Exception as e:
            logger.error(f"Error extracting CSV text: {e}")
            raise CSVError(f"Cannot read CSV file: {str(e)}")

    async def _store_csv_context(self, conversation_id: str, csv_text: str, csv_path: Path):
        """Store CSV context in Redis (simulating vector DB storage)"""
        try:
            key = f"csv_context:{conversation_id}"
            context_data = {
                "text": csv_text,
                "file_path": str(csv_path),
                "timestamp": datetime.now().isoformat()
            }
            await redis_service.set(key, json.dumps(context_data), expire=86400)
            logger.info(f"CSV context stored for {conversation_id}")
        except Exception as e:
            logger.error(f"Error storing CSV context: {e}")

    async def _get_csv_context(self, conversation_id: str) -> Optional[Dict]:
        """Retrieve CSV context from Redis"""
        try:
            key = f"csv_context:{conversation_id}"
            data = await redis_service.get(key)
            if data:
                return json.loads(data)
            return None
        except Exception as e:
            logger.error(f"Error getting CSV context: {e}")
            return None

    async def _build_rag_prompt(
        self,
        message: str,
        conversation_id: str,
        history: List[Dict]
    ) -> str:
        """Build prompt with RAG context"""
        # Get CSV context
        csv_context = await self._get_csv_context(conversation_id)
        
        if not csv_context:
            return message
        
        # Build conversation history
        history_text = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in history[-5:]  # Last 5 messages
        ])
        
        # Build full prompt
        prompt = f"""
You are analyzing a CSV file. Here is the data:

{csv_context['text']}

Conversation History:
{history_text}

Current User Question: {message}

Instructions:
- Answer the user's question based on the CSV data provided above
- If asking for statistics, calculate or describe them from the data shown
- If the data is incomplete (only showing first 20 rows), mention that the analysis is based on a preview
- Be precise and provide numbers when possible
- If you cannot calculate exact values, provide estimates and explain why

Your Response:
"""
        
        return prompt

    # ===== Original Pandas-based methods (for URL mode) =====
    
    async def _load_from_url(self, csv_url: str, csv_path: Path) -> Optional[pd.DataFrame]:
        """Load CSV from URL (original logic)"""
        try:
            # Validate URL
            if not csv_url.startswith(('http://', 'https://')):
                raise CSVError("Invalid URL format")
            
            # Auto-convert GitHub URLs
            if 'github.com' in csv_url and '/blob/' in csv_url:
                csv_url = csv_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                logger.info(f"Converted GitHub URL: {csv_url}")
            
            # Try master branch first, then main if fails
            if 'raw.githubusercontent.com' in csv_url and '/master/' in csv_url:
                original_url = csv_url
                try:
                    response = requests.get(csv_url, timeout=30)
                    response.raise_for_status()
                except requests.HTTPError as e:
                    if e.response.status_code == 404:
                        csv_url = csv_url.replace('/master/', '/main/')
                        logger.info(f"Trying with main branch: {csv_url}")
                        response = requests.get(csv_url, timeout=30)
                        response.raise_for_status()
                    else:
                        raise
            else:
                response = requests.get(csv_url, timeout=30)
                response.raise_for_status()
            
            # Parse CSV
            df = pd.read_csv(StringIO(response.content.decode('utf-8')))
            
            # Validate
            if df.empty:
                raise CSVError("CSV file is empty")
            
            # Save
            df.to_csv(csv_path, index=False)
            
            logger.info(f"CSV loaded from URL: {len(df)} rows, {len(df.columns)} columns")
            return df
            
        except Exception as e:
            logger.error(f"Error loading from URL: {e}")
            raise CSVError(f"Cannot load CSV from URL: {str(e)}")

    async def get_history(self, conversation_id: str) -> List[Dict[str, Any]]:
        try:
            key = f"csv_chat_history:{conversation_id}"
            data = await redis_service.get(key)
            return json.loads(data) if data else []
        except Exception as e:
            logger.error(f"Error getting history: {e}")
            return []

    async def save_history(self, conversation_id: str, history: List[Dict[str, Any]]):
        try:
            key = f"csv_chat_history:{conversation_id}"
            await redis_service.set(key, json.dumps(history), expire=86400)
        except Exception as e:
            logger.error(f"Error saving history: {e}")


csv_service = CSVService()