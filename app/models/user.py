from pydantic import BaseModel, EmailStr, Field
from typing import Optional
from datetime import datetime

class UserBase(BaseModel):
    """Base user model."""
    username: str = Field(..., min_length=3, max_length=50)
    email: EmailStr
    full_name: Optional[str] = None
    is_active: bool = True

class UserCreate(UserBase):
    """User creation model."""
    password: str = Field(..., min_length=8)

class UserUpdate(BaseModel):
    """User update model."""
    username: Optional[str] = Field(None, min_length=3, max_length=50)
    email: Optional[EmailStr] = None
    full_name: Optional[str] = None
    is_active: Optional[bool] = None

class UserInDB(UserBase):
    """User model for database operations."""
    id: int
    hashed_password: str
    created_at: datetime
    updated_at: datetime

class User(UserBase):
    """User model for API responses."""
    id: int
    created_at: datetime
    updated_at: datetime

class Token(BaseModel):
    """Token model for authentication."""
    access_token: str
    token_type: str = "bearer"
    expires_in: int

class TokenData(BaseModel):
    """Token data model."""
    username: Optional[str] = None

class UserLogin(BaseModel):
    """User login model."""
    username: str
    password: str
