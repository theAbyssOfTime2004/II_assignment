from typing import Optional, List, Dict, Any, Tuple
from datetime import datetime
import uuid
import json
import pandas as pd
from pathlib import Path
from llama_index.experimental.query_engine import PandasQueryEngine
from llama_index.llms.openai import OpenAI as LlamaOpenAI
from llama_index.core import Settings
from app.services.redis_service import redis_service
from app.core.config import OPENAI_API_KEY, OPENAI_MODEL_NAME
import logging
import requests
from io import StringIO

logger = logging.getLogger(__name__)

class CSVError(Exception):
    """Custom exception for CSV processing errors"""
    pass

class CSVService:
    MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
    MAX_ROWS = 1_000_000  # 1 million rows
    ALLOWED_EXTENSIONS = {'.csv', '.txt'}
    
    def __init__(self):
        self.upload_dir = Path("uploads/csv")
        self.upload_dir.mkdir(parents=True, exist_ok=True)
        
        # Configure llama-index to use gpt-4o-mini
        try:
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
        conversation_id = conversation_id or str(uuid.uuid4())
        
        try:
            # **NEW: Check for conflicting inputs**
            if csv_data and csv_url:
                return {
                    "response": "⚠️ **Lỗi:** Không thể xử lý đồng thời cả file upload và URL.\n\n💡 **Gợi ý:** Chỉ chọn một trong hai:\n- Upload file CSV trực tiếp, HOẶC\n- Cung cấp URL CSV",
                    "conversation_id": conversation_id
                }
        
            # Load or get DataFrame
            df = await self.get_or_load_dataframe(conversation_id, csv_data, csv_url)
            
            if df is None:
                return {
                    "response": "⚠️ **Lỗi:** Vui lòng upload file CSV hoặc cung cấp URL trước khi đặt câu hỏi.\n\n📄 **Định dạng hỗ trợ:** .csv, .txt\n📏 **Kích thước tối đa:** 50MB",
                    "conversation_id": conversation_id
                }
            
            # Load history
            history = await self.get_history(conversation_id)
            
            # Add user message
            user_msg = {
                "role": "user",
                "content": message,
                "timestamp": datetime.now().isoformat()
            }
            history.append(user_msg)
            
            # Process query
            response_text, chart_data = await self.query_dataframe(df, message)
            
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
                "response": f"⚠️ **Lỗi CSV:** {str(e)}",
                "conversation_id": conversation_id
            }
        except Exception as e:
            logger.error(f"Unexpected error in CSV processing: {e}")
            return {
                "response": "❌ **Lỗi không xác định.** Vui lòng thử lại hoặc upload file CSV khác.",
                "conversation_id": conversation_id
            }

    async def get_or_load_dataframe(
        self,
        conversation_id: str,
        csv_data: Optional[bytes] = None,
        csv_url: Optional[str] = None
    ) -> Optional[pd.DataFrame]:
        csv_path = self.upload_dir / f"{conversation_id}.csv"
        
        # If CSV already exists for this conversation
        if csv_path.exists():
            try:
                return pd.read_csv(csv_path)
            except Exception as e:
                logger.error(f"Error reading existing CSV: {e}")
                raise CSVError("File CSV đã lưu bị lỗi. Vui lòng upload lại.")
        
        # Load from bytes
        if csv_data:
            return await self._load_from_bytes(csv_data, csv_path)
        
        # Load from URL
        if csv_url:
            return await self._load_from_url(csv_url, csv_path)
        
        return None

    async def _load_from_bytes(self, csv_data: bytes, csv_path: Path) -> pd.DataFrame:
        """Load CSV from uploaded bytes with validation"""
        try:
            # Check file size
            if len(csv_data) > self.MAX_FILE_SIZE:
                raise CSVError(f"File quá lớn (max {self.MAX_FILE_SIZE // (1024*1024)}MB). Vui lòng upload file nhỏ hơn.")
            
            # Check if empty
            if len(csv_data) == 0:
                raise CSVError("File CSV trống. Vui lòng upload file có dữ liệu.")
            
            # Save to temp file
            with open(csv_path, "wb") as f:
                f.write(csv_data)
            
            # Try to read with multiple encodings
            encodings = ['utf-8', 'latin-1', 'iso-8859-1', 'cp1252']
            df = None
            last_error = None
            
            for encoding in encodings:
                try:
                    df = pd.read_csv(csv_path, encoding=encoding)
                    break
                except UnicodeDecodeError as e:
                    last_error = e
                    continue
            
            if df is None:
                raise CSVError(f"Không thể đọc file CSV. Encoding không hợp lệ: {last_error}")
            
            # Validate DataFrame
            return self._validate_dataframe(df, csv_path)
            
        except pd.errors.EmptyDataError:
            raise CSVError("File CSV không có dữ liệu.")
        except pd.errors.ParserError as e:
            raise CSVError(f"Lỗi parse CSV: {str(e)}. File có thể bị hỏng hoặc không đúng định dạng.")
        except Exception as e:
            if csv_path.exists():
                csv_path.unlink()  # Clean up
            raise CSVError(f"Lỗi khi đọc file: {str(e)}")

    async def _load_from_url(self, csv_url: str, csv_path: Path) -> Optional[pd.DataFrame]:
        """Load CSV from URL with validation"""
        try:
            # Validate URL format
            if not csv_url.startswith(('http://', 'https://')):
                raise CSVError("URL không hợp lệ. URL phải bắt đầu bằng http:// hoặc https://")
            
            # **NEW: Auto-convert GitHub URLs to raw URLs**
            if 'github.com' in csv_url and '/blob/' in csv_url:
                csv_url = csv_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                logger.info(f"Converted GitHub URL to raw URL: {csv_url}")
            
            # Check if URL ends with .csv
            if not any(csv_url.lower().endswith(ext) for ext in ['.csv', '.txt']):
                raise CSVError("URL phải kết thúc bằng .csv hoặc .txt\n\n💡 **Ví dụ:** https://example.com/data.csv")
            # Try to read with timeout and size limit
            from io import StringIO

            logger.info(f"Fetching CSV from URL: {csv_url}")
            
            response = requests.get(csv_url, timeout=30, stream=True)
            response.raise_for_status()
            
            # Check content type
            content_type = response.headers.get('content-type', '').lower()
            
            # **NEW: Better content-type validation with helpful error**
            if 'text/html' in content_type:
                logger.warning(f"Received HTML instead of CSV from: {csv_url}")
                error_msg = "⚠️ **URL trả về trang HTML thay vì file CSV.**\n\n"
                
                if 'github.com' in csv_url:
                    raw_url = csv_url.replace('github.com', 'raw.githubusercontent.com').replace('/blob/', '/')
                    error_msg += f"💡 **GitHub URL:** Sử dụng URL raw thay vì URL blob:\n\n"
                    error_msg += f"❌ **Sai:** `{csv_url}`\n\n"
                    error_msg += f"✅ **Đúng:** `{raw_url}`\n\n"
                    error_msg += f"**Hoặc:** Click nút 'Raw' trên trang GitHub để lấy URL đúng."
                else:
                    error_msg += "💡 **Gợi ý:**\n"
                    error_msg += "- Đảm bảo URL trỏ trực tiếp đến file CSV (không phải trang web)\n"
                    error_msg += "- Thử click chuột phải vào link download → Copy Link Address\n"
                    error_msg += "- Hoặc upload file CSV trực tiếp thay vì dùng URL"
                
                raise CSVError(error_msg)
            
            if 'text/csv' not in content_type and 'text/plain' not in content_type and 'application/octet-stream' not in content_type:
                logger.warning(f"Unexpected content type: {content_type}")
                raise CSVError(f"Content-Type không hỗ trợ: {content_type}\n\n✅ **Chấp nhận:** text/csv, text/plain, application/octet-stream\n\n💡 URL có thể không trỏ đến file CSV thực sự.")
            
            # Check file size from headers
            content_length = response.headers.get('content-length')
            if content_length and int(content_length) > self.MAX_FILE_SIZE:
                raise CSVError(f"File quá lớn ({int(content_length) / (1024*1024):.1f}MB). Max: {self.MAX_FILE_SIZE // (1024*1024)}MB")
            
            # Download with size limit
            content = b""
            for chunk in response.iter_content(chunk_size=8192):
                content += chunk
                if len(content) > self.MAX_FILE_SIZE:
                    raise CSVError(f"File quá lớn (>{self.MAX_FILE_SIZE // (1024*1024)}MB). Vui lòng upload trực tiếp hoặc dùng file nhỏ hơn.")
            
            # Parse CSV
            try:
                df = pd.read_csv(StringIO(content.decode('utf-8')))
            except UnicodeDecodeError:
                # Try other encodings
                encodings = ['latin-1', 'iso-8859-1', 'cp1252']
                df = None
                for encoding in encodings:
                    try:
                        df = pd.read_csv(StringIO(content.decode(encoding)))
                        logger.info(f"Successfully decoded with {encoding}")
                        break
                    except Exception:
                        continue
                
                if df is None:
                    raise CSVError("Không thể decode file CSV. Encoding không được hỗ trợ.\n\n💡 Thử lưu file với UTF-8 encoding và upload trực tiếp.")
            
            # Validate and save
            df = self._validate_dataframe(df, csv_path)
            df.to_csv(csv_path, index=False)
            
            return df
            
        except requests.exceptions.Timeout:
            raise CSVError("⏱️ **Timeout khi tải file từ URL.**\n\n💡 **Gợi ý:**\n- Thử lại sau\n- Upload file trực tiếp nếu file quá lớn\n- Kiểm tra kết nối mạng")
        except requests.exceptions.ConnectionError:
            raise CSVError("🔌 **Không thể kết nối đến URL.**\n\n💡 **Kiểm tra:**\n- URL có đúng không?\n- Kết nối internet có ổn định?\n- Server có đang hoạt động?")
        except requests.exceptions.HTTPError as e:
            if e.response.status_code == 404:
                raise CSVError("❌ **404 Not Found:** File không tồn tại.\n\n💡 Kiểm tra lại URL hoặc quyền truy cập.")
            elif e.response.status_code == 403:
                raise CSVError("🔒 **403 Forbidden:** Không có quyền truy cập.\n\n💡 File có thể là private hoặc cần authentication.")
            else:
                raise CSVError(f"❌ **HTTP {e.response.status_code}:** {str(e)}\n\n💡 Thử upload file trực tiếp.")
        except requests.exceptions.RequestException as e:
            raise CSVError(f"🌐 **Lỗi khi tải file từ URL:** {str(e)}\n\n💡 Thử upload file CSV trực tiếp thay vì dùng URL.")
        except pd.errors.EmptyDataError:
            raise CSVError("📭 **File CSV từ URL không có dữ liệu.**\n\n💡 Kiểm tra lại nội dung file.")
        except pd.errors.ParserError as e:
            raise CSVError(f"⚠️ **Lỗi parse CSV từ URL:** {str(e)}\n\n💡 **Có thể do:**\n- File không đúng định dạng CSV\n- URL trỏ đến trang HTML thay vì file CSV\n- File bị hỏng\n\n**Giải pháp:** Upload file trực tiếp để dễ xử lý hơn.")
        except Exception as e:
            logger.error(f"Unexpected error loading from URL: {e}")
            raise CSVError(f"❌ **Lỗi không xác định khi tải từ URL:** {str(e)}\n\n💡 **Khuyến nghị:** Upload file CSV trực tiếp sẽ ổn định hơn.")

    def _validate_dataframe(self, df: pd.DataFrame, csv_path: Path) -> pd.DataFrame:
        """Validate DataFrame and apply constraints"""
        try:
            # Check if empty
            if df.empty:
                raise CSVError("DataFrame rỗng. File CSV không có dữ liệu.")
            
            # Check number of rows
            if len(df) > self.MAX_ROWS:
                logger.warning(f"DataFrame has {len(df)} rows, truncating to {self.MAX_ROWS}")
                df = df.head(self.MAX_ROWS)
            
            # Check if has columns
            if len(df.columns) == 0:
                raise CSVError("File CSV không có cột nào.")
            
            # Clean column names
            df.columns = df.columns.str.strip()
            
            # Check for all-null columns
            null_cols = df.columns[df.isnull().all()].tolist()
            if null_cols:
                logger.warning(f"Dropping all-null columns: {null_cols}")
                df = df.drop(columns=null_cols)
            
            # Log DataFrame info
            logger.info(f"DataFrame loaded successfully: {len(df)} rows, {len(df.columns)} columns")
            logger.info(f"Columns: {list(df.columns)}")
            
            return df
            
        except Exception as e:
            if csv_path.exists():
                csv_path.unlink()
            raise CSVError(f"Lỗi validate DataFrame: {str(e)}")

    async def query_dataframe(self, df: pd.DataFrame, query: str) -> Tuple[str, Optional[dict]]:
        """Query DataFrame with error handling"""
        chart_data = None
        
        try:
            # Check for common queries
            if "summarize" in query.lower() or "tóm tắt" in query.lower():
                response = self.summarize_dataset(df)
            elif "stats" in query.lower() or "thống kê" in query.lower():
                response = self.basic_stats(df)
            elif "missing" in query.lower() or "thiếu" in query.lower():
                response = self.missing_values(df)
            elif "histogram" in query.lower() or "biểu đồ" in query.lower():
                response, chart_data = self.create_histogram(df, query)
            else:
                # Use PandasQueryEngine for complex queries
                try:
                    query_engine = PandasQueryEngine(
                        df=df, 
                        verbose=True,
                        synthesize_response=True
                    )
                    result = query_engine.query(query)
                    response = str(result)
                except Exception as e:
                    logger.error(f"PandasQueryEngine error: {e}")
                    # Fallback to simple description
                    response = f"⚠️ Không thể xử lý câu hỏi phức tạp.\n\n"
                    response += self.summarize_dataset(df)
                    response += "\n\n💡 **Gợi ý:** Thử các câu hỏi như:\n"
                    response += "- `summarize` - Tóm tắt dataset\n"
                    response += "- `stats` - Thống kê cơ bản\n"
                    response += "- `missing` - Giá trị thiếu\n"
                    response += f"- `histogram [tên cột]` - Biểu đồ (Ví dụ: `histogram Age`)"
            
            return response, chart_data
            
        except Exception as e:
            logger.error(f"Error in query_dataframe: {e}")
            return f"❌ Lỗi khi xử lý câu hỏi: {str(e)}", None

    def summarize_dataset(self, df: pd.DataFrame) -> str:
        """Create a well-formatted dataset summary"""
        try:
            # Get data types info
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
            
            summary = "## 📊 Tóm Tắt Dataset\n\n"
            
            # Basic info
            summary += "### 📈 Thông tin cơ bản\n"
            summary += f"- **Tổng số dòng:** {len(df):,}\n"
            summary += f"- **Tổng số cột:** {len(df.columns)}\n"
            summary += f"- **Cột số:** {len(numeric_cols)}\n"
            summary += f"- **Cột phân loại:** {len(categorical_cols)}\n"
            summary += f"- **Dung lượng bộ nhớ:** {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB\n\n"
            
            # Column list with types
            summary += "### 📋 Danh sách cột\n"
            summary += "| # | Tên cột | Kiểu dữ liệu | Giá trị null |\n"
            summary += "|---|---------|-------------|-------------|\n"
            for idx, col in enumerate(df.columns, 1):
                dtype = str(df[col].dtype)
                null_count = df[col].isnull().sum()
                null_pct = f"{null_count/len(df)*100:.1f}%" if null_count > 0 else "0%"
                summary += f"| {idx} | `{col}` | {dtype} | {null_count:,} ({null_pct}) |\n"
            
            summary += "\n"
            
            # Preview data (formatted as table)
            summary += "### 👀 Xem trước dữ liệu (5 dòng đầu)\n\n"
            summary += self._format_dataframe_as_table(df.head())
            
            return summary
            
        except Exception as e:
            logger.error(f"Error in summarize_dataset: {e}")
            return f"⚠️ Lỗi khi tóm tắt: {str(e)}"

    def _format_dataframe_as_table(self, df: pd.DataFrame, max_rows: int = 10) -> str:
        """Format DataFrame as Markdown table with better formatting"""
        try:
            # Limit number of rows
            df_display = df.head(max_rows)
            
            # Create header
            table = "| " + " | ".join(df_display.columns) + " |\n"
            table += "|" + "|".join(["---"] * len(df_display.columns)) + "|\n"
            
            # Add rows
            for _, row in df_display.iterrows():
                formatted_row = []
                for val in row:
                    # Format values nicely
                    if pd.isna(val):
                        formatted_row.append("*null*")
                    elif isinstance(val, (int, float)):
                        if isinstance(val, float):
                            formatted_row.append(f"{val:,.2f}")
                        else:
                            formatted_row.append(f"{val:,}")
                    else:
                        # Truncate long strings
                        str_val = str(val)
                        if len(str_val) > 30:
                            str_val = str_val[:27] + "..."
                        formatted_row.append(str_val)
                
                table += "| " + " | ".join(formatted_row) + " |\n"
            
            return table
            
        except Exception as e:
            logger.error(f"Error formatting table: {e}")
            return f"```\n{df.to_string()}\n```"

    def basic_stats(self, df: pd.DataFrame) -> str:
        """Create well-formatted statistics summary"""
        try:
            numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
            
            if len(numeric_cols) == 0:
                return "⚠️ **Không có cột số nào** trong dataset để tính thống kê."
            
            stats = "## 📈 Thống Kê Cơ Bản\n\n"
            
            for col in numeric_cols:
                stats += f"### 📊 {col}\n\n"
                
                col_data = df[col].dropna()
                
                stats += "| Chỉ số | Giá trị |\n"
                stats += "|--------|--------|\n"
                stats += f"| **Count** | {len(col_data):,} |\n"
                stats += f"| **Mean** | {col_data.mean():,.2f} |\n"
                stats += f"| **Std Dev** | {col_data.std():,.2f} |\n"
                stats += f"| **Min** | {col_data.min():,.2f} |\n"
                stats += f"| **25%** | {col_data.quantile(0.25):,.2f} |\n"
                stats += f"| **Median (50%)** | {col_data.median():,.2f} |\n"
                stats += f"| **75%** | {col_data.quantile(0.75):,.2f} |\n"
                stats += f"| **Max** | {col_data.max():,.2f} |\n"
                stats += f"| **Null values** | {df[col].isnull().sum():,} |\n\n"
            
            return stats
            
        except Exception as e:
            logger.error(f"Error in basic_stats: {e}")
            return f"⚠️ Lỗi khi tính thống kê: {str(e)}"

    def missing_values(self, df: pd.DataFrame) -> str:
        """Create well-formatted missing values report"""
        try:
            missing = df.isnull().sum()
            missing = missing[missing > 0].sort_values(ascending=False)
            
            if len(missing) == 0:
                return "✅ **Không có giá trị thiếu** trong dataset!\n\n🎉 Dữ liệu hoàn chỉnh và sẵn sàng để phân tích."
            
            result = "## ⚠️ Báo Cáo Giá Trị Thiếu\n\n"
            result += f"**Tổng quan:** Có **{len(missing)}** cột chứa giá trị null.\n\n"
            
            result += "| Tên cột | Số null | Tỷ lệ | Biểu đồ |\n"
            result += "|---------|---------|-------|--------|\n"
            
            for col, count in missing.items():
                pct = count / len(df) * 100
                # Create simple text progress bar
                bar_length = int(pct / 5)  # Scale to 20 chars max
                bar = "█" * bar_length + "░" * (20 - bar_length)
                result += f"| `{col}` | {count:,} | {pct:.1f}% | {bar} |\n"
            
            result += f"\n**💡 Gợi ý:** Xem xét xử lý giá trị thiếu trước khi phân tích sâu hơn."
            
            return result
            
        except Exception as e:
            logger.error(f"Error in missing_values: {e}")
            return f"⚠️ Lỗi khi kiểm tra giá trị thiếu: {str(e)}"

    def create_histogram(self, df: pd.DataFrame, query: str) -> Tuple[str, Optional[dict]]:
        """Create histogram with better formatting"""
        try:
            # Extract column name from query
            words = query.lower().split()
            column = None
            for word in words:
                if word in [col.lower() for col in df.columns]:
                    column = [col for col in df.columns if col.lower() == word][0]
                    break
            
            if not column or column not in df.columns:
                avail_numeric = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
                return f"⚠️ **Không tìm thấy cột:** `{query}`\n\n**Các cột số khả dụng:**\n" + "\n".join([f"- `{col}`" for col in avail_numeric]), None
            
            if not pd.api.types.is_numeric_dtype(df[column]):
                return f"⚠️ Cột `{column}` không phải là số. Không thể tạo histogram.", None
            
            # Create histogram data
            col_data = df[column].dropna()
            hist_data = col_data.value_counts().sort_index().head(50).to_dict()
            chart_data = {
                "type": "histogram",
                "column": column,
                "data": {str(k): int(v) for k, v in hist_data.items()}
            }
            
            response = f"## 📊 Histogram: `{column}`\n\n"
            
            # Statistics table
            response += "### 📈 Thống kê\n\n"
            response += "| Chỉ số | Giá trị |\n"
            response += "|--------|--------|\n"
            response += f"| **Count** | {len(col_data):,} |\n"
            response += f"| **Min** | {col_data.min():,.2f} |\n"
            response += f"| **Max** | {col_data.max():,.2f} |\n"
            response += f"| **Mean** | {col_data.mean():,.2f} |\n"
            response += f"| **Median** | {col_data.median():,.2f} |\n"
            response += f"| **Std Dev** | {col_data.std():,.2f} |\n\n"
            
            # Distribution info
            response += "### 📉 Phân phối\n\n"
            response += f"Hiển thị **{len(hist_data)}** giá trị duy nhất (unique values).\n\n"
            
            return response, chart_data
            
        except Exception as e:
            logger.error(f"Error in create_histogram: {e}")
            return f"⚠️ Lỗi khi tạo histogram: {str(e)}", None

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