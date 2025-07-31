"""
Document models and schemas
"""

from dataclasses import dataclass, asdict
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

class DocumentType(str, Enum):
    """Supported document types"""
    PDF = "pdf"
    DOCX = "docx"
    TXT = "txt"
    HTML = "html"
    EMAIL = "eml"

class Domain(str, Enum):
    """Business domains"""
    INSURANCE = "insurance"
    LEGAL = "legal"
    HR = "hr"
    COMPLIANCE = "compliance"
    GENERAL = "general"

@dataclass
class DocumentMetadata:
    """Document metadata structure"""
    doc_id: str
    filename: str
    doc_type: DocumentType
    upload_timestamp: datetime
    file_size: int
    content_hash: str
    domain: Domain
    page_count: Optional[int] = None
    word_count: Optional[int] = None
    language: Optional[str] = "en"
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        data = asdict(self)
        data['upload_timestamp'] = self.upload_timestamp.isoformat()
        return data

@dataclass
class ProcessedChunk:
    """Processed document chunk"""
    chunk_id: str
    doc_id: str
    content: str
    embedding: List[float]
    chunk_index: int
    metadata: Dict[str, Any]
    page_number: Optional[int] = None
    start_char: Optional[int] = None
    end_char: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)

class DocumentUploadRequest(BaseModel):
    """Document upload request model"""
    domain: Domain = Field(default=Domain.GENERAL, description="Business domain")
    language: str = Field(default="en", description="Document language")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class DocumentUploadResponse(BaseModel):
    """Document upload response model"""
    doc_id: str = Field(description="Generated document ID")
    filename: str = Field(description="Original filename")
    message: str = Field(description="Success message")
    chunks_count: int = Field(description="Number of chunks created")
    processing_time: float = Field(description="Processing time in seconds")

class DocumentInfo(BaseModel):
    """Document information model"""
    doc_id: str
    filename: str
    doc_type: DocumentType
    domain: Domain
    upload_timestamp: datetime
    file_size: int
    chunk_count: int
    word_count: Optional[int] = None
    page_count: Optional[int] = None

class ChunkInfo(BaseModel):
    """Chunk information model"""
    chunk_id: str
    doc_id: str
    content: str
    chunk_index: int
    similarity_score: Optional[float] = None
    page_number: Optional[int] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class DocumentSearchRequest(BaseModel):
    """Document search request"""
    query: str = Field(description="Search query")
    domain: Optional[Domain] = Field(default=None, description="Filter by domain")
    doc_types: Optional[List[DocumentType]] = Field(default=None, description="Filter by document types")
    max_results: int = Field(default=10, description="Maximum number of results")
    similarity_threshold: float = Field(default=0.5, description="Minimum similarity threshold")

class DocumentSearchResponse(BaseModel):
    """Document search response"""
    query: str
    results: List[ChunkInfo]
    total_found: int
    processing_time: float
