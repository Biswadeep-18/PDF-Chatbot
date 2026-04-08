from motor.motor_asyncio import AsyncIOMotorClient
from ..core.config import settings
from typing import Optional, Dict, Any

class MongoDBService:
    def __init__(self):
        self.client: Optional[AsyncIOMotorClient] = None
        self.db = None
        self.users_collection = None

    def connect(self):
        self.client = AsyncIOMotorClient(settings.MONGODB_URI)
        self.db = self.client[settings.DATABASE_NAME]
        self.users_collection = self.db.users
        print(f"Connected to MongoDB: {settings.MONGODB_URI}")

    async def get_user_by_username(self, username: str) -> Optional[Dict[str, Any]]:
        return await self.users_collection.find_one({"username": username})

    async def get_user_by_email(self, email: str) -> Optional[Dict[str, Any]]:
        return await self.users_collection.find_one({"email": email})

    async def create_user(self, user_data: Dict[str, Any]) -> str:
        # Set default fields for profile
        if "profile_image" not in user_data: user_data["profile_image"] = None
        if "theme" not in user_data: user_data["theme"] = "light"
        
        result = await self.users_collection.insert_one(user_data)
        return str(result.inserted_id)

    async def update_user(self, username: str, update_data: Dict[str, Any]) -> bool:
        result = await self.users_collection.update_one(
            {"username": username},
            {"$set": update_data}
        )
        return result.modified_count > 0

db_service = MongoDBService()
