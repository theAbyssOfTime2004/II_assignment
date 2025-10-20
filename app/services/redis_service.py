from redis import asyncio as aioredis
from app.core.config import REDIS_URL

class RedisService:
    def __init__(self):
        self.client = aioredis.Redis.from_url(REDIS_URL)

    async def get(self, key: str) -> str:
        return await self.client.get(key)

    async def set(self, key: str, value: str, expire: int = 3600):
        await self.client.set(key, value, ex=expire)

redis_service = RedisService()