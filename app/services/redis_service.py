from redis import asyncio as aioredis
from app.core.config import REDIS_URL
import logging

logger = logging.getLogger(__name__)

class RedisService:
    def __init__(self):
        try:
            self.client = aioredis.Redis.from_url(REDIS_URL, decode_responses=True)
            logger.info(f"Redis client initialized: {REDIS_URL}")
        except Exception as e:
            logger.error(f"Failed to initialize Redis: {e}")
            self.client = None

    async def get(self, key: str) -> str:
        try:
            if not self.client:
                logger.warning("Redis client not available")
                return None
            return await self.client.get(key)
        except Exception as e:
            logger.error(f"Redis GET error for key {key}: {e}")
            return None

    async def set(self, key: str, value: str, expire: int = 3600):
        try:
            if not self.client:
                logger.warning("Redis client not available")
                return
            await self.client.set(key, value, ex=expire)
        except Exception as e:
            logger.error(f"Redis SET error for key {key}: {e}")

    async def delete(self, key: str):
        try:
            if not self.client:
                logger.warning("Redis client not available")
                return
            await self.client.delete(key)
        except Exception as e:
            logger.error(f"Redis DELETE error for key {key}: {e}")

    async def ping(self) -> bool:
        """Check if Redis is available"""
        try:
            if not self.client:
                return False
            await self.client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis ping failed: {e}")
            return False

redis_service = RedisService()