"""
Query models and schemas
"""

from dataclasses import dataclass
from datetime import datetime
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from enum import Enum

from .document import Domain, ChunkInfo

class QueryType(str, Enum):
    """Types of queries"""
    FACTUAL = "factual"
    COMPARATIVE = "comparative"
    ANALYTICAL = "analytical"
    PROCEDURAL = "procedural"
    COMPLIANCE = "compliance"

class ProcessingStatus(str, Enum):
    """Query processing status"""
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"

@dataclass
class QueryResult:
    """Query processing result"""
    query: str
    matched_chunks: List[Dict[str, Any]]
    confidence_score: float
    decision_rationale: str
    structured_response: Dict[str, Any]
    processing_time: float
    query_type: Optional[QueryType] = None
    domain: Optional[Domain] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'query': self.query,
            'matched_chunks': self.matched_chunks,
            'confidence_score': self.confidence_score,
            'decision_rationale': self.decision_rationale,
            'structured_response': self.structured_response,
            'processing_time': self.processing_time,
            'query_type': self.query_type,
            'domain': self.domain
        }

class QueryRequest(BaseModel):
    """API Query request model"""
    documents: List[str] = Field(description="Document URLs or IDs")
    questions: List[str] = Field(description="List of questions to process")
    domain: Optional[Domain] = Field(default=None, description="Specific domain context")
    max_chunks: int = Field(default=10, description="Maximum chunks to retrieve per query")
    include_rationale: bool = Field(default=True, description="Include decision rationale")

class QueryResponse(BaseModel):
    """API Query response model"""
    answers: List[str] = Field(description="List of answers corresponding to questions")

class DetailedQueryRequest(BaseModel):
    """Detailed query request with additional parameters"""
    query: str = Field(description="The query to process")
    documents: Optional[List[str]] = Field(default=None, description="Specific document IDs to search")
    domain: Optional[Domain] = Field(default=None, description="Domain context")
    query_type: Optional[QueryType] = Field(default=None, description="Type of query")
    max_chunks: int = Field(default=10, description="Maximum chunks to retrieve")
    similarity_threshold: float = Field(default=0.5, description="Minimum similarity threshold")
    include_metadata: bool = Field(default=True, description="Include chunk metadata")
    include_rationale: bool = Field(default=True, description="Include decision rationale")
    temperature: float = Field(default=0.1, description="LLM temperature for response generation")

class DetailedQueryResponse(BaseModel):
    """Detailed query response with full information"""
    query: str
    answer: str
    confidence_score: float
    processing_time: float
    matched_chunks: List[ChunkInfo]
    decision_rationale: Optional[str] = None
    query_type: Optional[QueryType] = None
    domain: Optional[Domain] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

class BulkQueryRequest(BaseModel):
    """Bulk query processing request"""
    queries: List[DetailedQueryRequest] = Field(description="List of queries to process")
    parallel_processing: bool = Field(default=True, description="Process queries in parallel")
    max_concurrent: int = Field(default=5, description="Maximum concurrent queries")

class BulkQueryResponse(BaseModel):
    """Bulk query processing response"""
    results: List[DetailedQueryResponse]
    total_queries: int
    successful_queries: int
    failed_queries: int
    total_processing_time: float

class QueryAnalytics(BaseModel):
    """Query analytics and metrics"""
    query_id: str
    query_text: str
    domain: Optional[Domain]
    query_type: Optional[QueryType]
    confidence_score: float
    processing_time: float
    chunks_retrieved: int
    timestamp: datetime
    user_feedback: Optional[float] = None  # User rating 1-5
    
class QueryHistory(BaseModel):
    """Query history response"""
    queries: List[QueryAnalytics]
    total_count: int
    average_confidence: float
    average_processing_time: float
    domain_distribution: Dict[str, int]
