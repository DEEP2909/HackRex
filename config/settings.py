"""
Configuration settings for the LLM Query Retrieval System
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional
import os

class Settings(BaseSettings):
    """Application settings"""
    
    # Server configuration
    host: str = Field(default="0.0.0.0", description="Host to bind the server")
    port: int = Field(default=8000, description="Port to bind the server")
    debug: bool = Field(default=False, description="Enable debug mode")
    log_level: str = Field(default="INFO", description="Logging level")
    
    # Database configuration
    database_url: str = Field(
        default="sqlite:///./document_retrieval.db",
        description="Database connection URL"
    )
    
    # Redis configuration
    redis_url: Optional[str] = Field(
        default=None,
        description="Redis connection URL for caching"
    )
    
    # API Keys
    openai_api_key: Optional[str] = Field(
        default=None,
        description="OpenAI API key"
    )
    
    anthropic_api_key: Optional[str] = Field(
        default=None,
        description="Anthropic API key"
    )
    
    pinecone_api_key: Optional[str] = Field(
        default=None,
        description="Pinecone API key"
    )
    
    pinecone_env: str = Field(
        default="us-west1-gcp",
        description="Pinecone environment"
    )
    
    # Model configuration
    embedding_model: str = Field(
        default="all-MiniLM-L6-v2",
        description="Sentence transformer model for embeddings"
    )
    
    llm_model: str = Field(
        default="gpt-4",
        description="LLM model for query processing"
    )
    
    # Processing configuration
    max_chunk_size: int = Field(
        default=500,
        description="Maximum chunk size for document processing"
    )
    
    max_chunks_per_query: int = Field(
        default=10,
        description="Maximum chunks to retrieve per query"
    )
    
    similarity_threshold: float = Field(
        default=0.5,
        description="Minimum similarity threshold for chunk retrieval"
    )
    
    # File upload configuration
    max_file_size: int = Field(
        default=10 * 1024 * 1024,  # 10MB
        description="Maximum file size for uploads"
    )
    
    allowed_file_types: list = Field(
        default=[".pdf", ".docx", ".txt", ".html", ".eml"],
        description="Allowed file types for upload"
    )
    
    # Storage paths
    upload_path: str = Field(
        default="./data/temp",
        description="Temporary upload directory"
    )
    
    log_path: str = Field(
        default="./logs",
        description="Log file directory"
    )
    
    # Security
    secret_key: str = Field(
        default="your-secret-key-change-in-production",
        description="Secret key for JWT tokens"
    )
    
    access_token_expire_minutes: int = Field(
        default=30,
        description="Access token expiration time in minutes"
    )
    
    # CORS settings
    cors_origins: list = Field(
        default=["*"],
        description="Allowed CORS origins"
    )
    
    # Rate limiting
    rate_limit_requests: int = Field(
        default=100,
        description="Number of requests allowed per window"
    )
    
    rate_limit_window: int = Field(
        default=60,
        description="Rate limit window in seconds"
    )
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = False
        
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        
        # Create directories if they don't exist
        os.makedirs(self.upload_path, exist_ok=True)
        os.makedirs(self.log_path, exist_ok=True)
        
        # Validate API keys
        if not self.openai_api_key and not self.anthropic_api_key:
            raise ValueError("At least one LLM API key must be provided")

# Global settings instance
settings = Settings()
