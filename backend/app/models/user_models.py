from pydantic import BaseModel, EmailStr
from typing import Optional

class UserCreate(BaseModel):
    username: str
    email: EmailStr
    password: str
    
class UserLogin(BaseModel):
    username: str
    password: str
    
class UserResponse(BaseModel):
    id: str
    username: str
    email: str
    profile_image: Optional[str] = None
    theme: Optional[str] = "light"

class ProfileUpdate(BaseModel):
    theme: Optional[str] = None
    profile_image: Optional[str] = None
