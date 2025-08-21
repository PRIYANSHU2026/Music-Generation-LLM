from pydantic_settings import BaseSettings
from typing import List
import os

class Settings(BaseSettings):
    # API Configuration
    API_V1_STR: str = "/api/v1"
    PROJECT_NAME: str = "Music Generation LLM"
    
    # Server Configuration
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    DEBUG: bool = False
    
    # Security Configuration
    SECRET_KEY: str = "your-secret-key-change-in-production"
    ALGORITHM: str = "HS256"
    ACCESS_TOKEN_EXPIRE_MINUTES: int = 30
    
    # CORS Configuration
    ALLOWED_ORIGINS: List[str] = ["*"]
    
    # Database Configuration
    DATABASE_URL: str = "sqlite:///./music_llm.db"
    
    # LLM Configuration
    LLM_MODEL_NAME: str = "gpt-3.5-turbo"
    LLM_API_KEY: str = ""
    LLM_MAX_TOKENS: int = 1000
    LLM_TEMPERATURE: float = 0.7
    
    # Music Generation Configuration
    MAX_MUSIC_LENGTH: int = 300  # seconds
    SUPPORTED_GENRES: List[str] = ["pop", "rock", "jazz", "classical", "electronic"]
    OUTPUT_FORMATS: List[str] = ["midi", "wav", "mp3"]
    
    # Rate Limiting
    RATE_LIMIT_PER_MINUTE: int = 60
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    class Config:
        env_file = ".env"
        case_sensitive = True

# Create settings instance
settings = Settings()

# Override with environment variables if present
settings.SECRET_KEY = os.getenv("SECRET_KEY", settings.SECRET_KEY)
settings.LLM_API_KEY = os.getenv("LLM_API_KEY", settings.LLM_API_KEY)
settings.DATABASE_URL = os.getenv("DATABASE_URL", settings.DATABASE_URL)
settings.DEBUG = os.getenv("DEBUG", "false").lower() == "true"
