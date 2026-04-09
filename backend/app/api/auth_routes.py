from fastapi import APIRouter, Depends, HTTPException, status, Header
from typing import Optional
from ..models.user_models import UserCreate, UserLogin, UserResponse
from ..services.mongodb_service import db_service
from ..core.auth_utils import get_password_hash, verify_password, create_access_token, decode_access_token

router = APIRouter(prefix="/auth", tags=["Authentication"])

@router.post("/register")
async def register(user: UserCreate):
    existing_user = await db_service.get_user_by_username(user.username)
    if existing_user:
        raise HTTPException(status_code=400, detail="Username already registered")
        
    existing_email = await db_service.get_user_by_email(user.email)
    if existing_email:
        raise HTTPException(status_code=400, detail="Email already registered")
        
    hashed_password = get_password_hash(user.password)
    user_dict = dict(user)
    user_dict["password"] = hashed_password
    
    user_id = await db_service.create_user(user_dict)
    return {"message": "User registered successfully", "id": user_id}

@router.post("/login")
async def login(user: UserLogin):
    db_user = await db_service.get_user_by_username(user.username)
    if not db_user or not verify_password(user.password, db_user["password"]):
        raise HTTPException(status_code=401, detail="Invalid credentials")
        
    access_token = create_access_token(data={"sub": db_user["username"]})
    return {"access_token": access_token, "token_type": "bearer"}
async def get_current_user(authorization: Optional[str] = Header(None)):
    if not authorization or not authorization.startswith("Bearer "):
        raise HTTPException(status_code=401, detail="Invalid authorization header")
    
    token = authorization.split(" ")[1]
    payload = decode_access_token(token)
    if not payload:
        raise HTTPException(status_code=401, detail="Invalid token")
    
    username = payload.get("sub")
    if not username:
        raise HTTPException(status_code=401, detail="Invalid token payload")
        
    user = await db_service.get_user_by_username(username)
    if not user:
        raise HTTPException(status_code=401, detail="User not found")
    
    # Remove password from response
    if "password" in user:
        del user["password"]
    if "_id" in user:
        user["id"] = str(user["_id"])
        del user["_id"]
        
    return user

@router.get("/me")
async def get_me(current_user: dict = Depends(get_current_user)):
    return current_user

@router.put("/profile")
async def update_profile(data: dict, current_user: dict = Depends(get_current_user)):
    success = await db_service.update_user(current_user["username"], data)
    if not success:
        raise HTTPException(status_code=400, detail="Update failed")
    
    updated_user = await db_service.get_user_by_username(current_user["username"])
    if "password" in updated_user:
        del updated_user["password"]
    if "_id" in updated_user:
        updated_user["id"] = str(updated_user["_id"])
        del updated_user["_id"]
        
    return updated_user
