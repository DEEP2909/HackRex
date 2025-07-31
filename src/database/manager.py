"""
Database manager for storing documents, chunks, and query analytics
"""

import json
import asyncio
import hashlib
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional
import sqlite3
import aiosqlite
from sqlalchemy import create_engine, text
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker

from ..models.document import DocumentMetadata, ProcessedChunk, DocumentInfo, DocumentType, Domain
from ..models.query import QueryResult, QueryAnalytics, QueryHistory
from ..utils.logger import get_logger
from config.settings import Settings

logger = get_logger(__name__)

class DatabaseManager:
    """Manages document and query storage with async support"""
    
    def __init__(self, db_url: str = None):
        self.settings = Settings()
        self.db_url = db_url or self.settings.database_url
        
        # Determine if using async or sync engine
        self.is_async = "sqlite+aiosqlite" in self.db_url or "postgresql+asyncpg" in self.db_url
        
        if self.is_async:
            self.engine = create_async_engine(self.db_url, echo=False)
            self.SessionLocal = sessionmaker(
                self.engine, class_=AsyncSession, expire_on_commit=False
            )
        else:
            self.engine = create_engine(self.db_url, echo=False)
            self.SessionLocal = sessionmaker(bind=self.engine)
    
    async def initialize(self):
        """Initialize database tables"""
        try:
            logger.info("Initializing database...")
            
            if self.is_async:
                async with self.engine.begin() as conn:
                    await self._create_tables_async(conn)
            else:
                with self.engine.begin() as conn:
                    self._create_tables_sync(conn)
            
            logger.info("Database initialized successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize database: {str(e)}")
            raise
    
    async def _create_tables_async(self, conn):
        """Create tables using async connection"""
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                upload_timestamp TIMESTAMP NOT NULL,
                file_size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                domain TEXT NOT NULL,
                page_count INTEGER,
                word_count INTEGER,
                language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                embedding_vector TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                domain TEXT,
                query_type TEXT,
                confidence_score REAL,
                processing_time REAL,
                matched_chunks_count INTEGER,
                user_feedback REAL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        await conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                similarity_score REAL,
                FOREIGN KEY (query_id) REFERENCES queries (query_id) ON DELETE CASCADE,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE
            )
        """))
        
        # Create indexes for better performance
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents (domain)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries (timestamp)"))
        await conn.execute(text("CREATE INDEX IF NOT EXISTS idx_queries_domain ON queries (domain)"))
        
        await conn.commit()
    
    def _create_tables_sync(self, conn):
        """Create tables using sync connection"""
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS documents (
                doc_id TEXT PRIMARY KEY,
                filename TEXT NOT NULL,
                doc_type TEXT NOT NULL,
                upload_timestamp TIMESTAMP NOT NULL,
                file_size INTEGER NOT NULL,
                content_hash TEXT NOT NULL,
                domain TEXT NOT NULL,
                page_count INTEGER,
                word_count INTEGER,
                language TEXT DEFAULT 'en',
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS chunks (
                chunk_id TEXT PRIMARY KEY,
                doc_id TEXT NOT NULL,
                content TEXT NOT NULL,
                chunk_index INTEGER NOT NULL,
                page_number INTEGER,
                start_char INTEGER,
                end_char INTEGER,
                embedding_vector TEXT,
                metadata TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (doc_id) REFERENCES documents (doc_id) ON DELETE CASCADE
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS queries (
                query_id TEXT PRIMARY KEY,
                query_text TEXT NOT NULL,
                domain TEXT,
                query_type TEXT,
                confidence_score REAL,
                processing_time REAL,
                matched_chunks_count INTEGER,
                user_feedback REAL,
                timestamp TIMESTAMP NOT NULL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """))
        
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS query_chunks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                query_id TEXT NOT NULL,
                chunk_id TEXT NOT NULL,
                similarity_score REAL,
                FOREIGN KEY (query_id) REFERENCES queries (query_id) ON DELETE CASCADE,
                FOREIGN KEY (chunk_id) REFERENCES chunks (chunk_id) ON DELETE CASCADE
            )
        """))
        
        # Create indexes
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_documents_domain ON documents (domain)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_chunks_doc_id ON chunks (doc_id)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_queries_timestamp ON queries (timestamp)"))
        conn.execute(text("CREATE INDEX IF NOT EXISTS idx_queries_domain ON queries (domain)"))
        
        conn.commit()
    
    async def store_document(self, doc_metadata: DocumentMetadata):
        """Store document metadata"""
        try:
            if self.is_async:
                async with self.engine.begin() as conn:
                    await conn.execute(text("""
                        INSERT OR REPLACE INTO documents 
                        (doc_id, filename, doc_type, upload_timestamp, file_size, content_hash, 
                         domain, page_count, word_count, language, updated_at)
                        VALUES (:doc_id, :filename, :doc_type, :upload_timestamp, :file_size, 
                               :content_hash, :domain, :page_count, :word_count, :language, :updated_at)
                    """), {
                        'doc_id': doc_metadata.doc_id,
                        'filename': doc_metadata.filename,
                        'doc_type': doc_metadata.doc_type.value,
                        'upload_timestamp': doc_metadata.upload_timestamp,
                        'file_size': doc_metadata.file_size,
                        'content_hash': doc_metadata.content_hash,
                        'domain': doc_metadata.domain.value,
                        'page_count': doc_metadata.page_count,
                        'word_count': doc_metadata.word_count,
                        'language': doc_metadata.language,
                        'updated_at': datetime.now()
                    })
            else:
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        INSERT OR REPLACE INTO documents 
                        (doc_id, filename, doc_type, upload_timestamp, file_size, content_hash, 
                         domain, page_count, word_count, language, updated_at)
                        VALUES (:doc_id, :filename, :doc_type, :upload_timestamp, :file_size, 
                               :content_hash, :domain, :page_count, :word_count, :language, :updated_at)
                    """), {
                        'doc_id': doc_metadata.doc_id,
                        'filename': doc_metadata.filename,
                        'doc_type': doc_metadata.doc_type.value,
                        'upload_timestamp': doc_metadata.upload_timestamp,
                        'file_size': doc_metadata.file_size,
                        'content_hash': doc_metadata.content_hash,
                        'domain': doc_metadata.domain.value,
                        'page_count': doc_metadata.page_count,
                        'word_count': doc_metadata.word_count,
                        'language': doc_metadata.language,
                        'updated_at': datetime.now()
                    })
            
            logger.info(f"Stored document metadata: {doc_metadata.doc_id}")
            
        except Exception as e:
            logger.error(f"Error storing document: {str(e)}")
            raise
    
    async def store_chunks(self, chunks: List[ProcessedChunk]):
        """Store document chunks"""
        try:
            if not chunks:
                return
            
            chunk_data = []
            for chunk in chunks:
                chunk_data.append({
                    'chunk_id': chunk.chunk_id,
                    'doc_id': chunk.doc_id,
                    'content': chunk.content,
                    'chunk_index': chunk.chunk_index,
                    'page_number': chunk.page_number,
                    'start_char': chunk.start_char,
                    'end_char': chunk.end_char,
                    'embedding_vector': json.dumps(chunk.embedding) if chunk.embedding else None,
                    'metadata': json.dumps(chunk.metadata)
                })
            
            if self.is_async:
                async with self.engine.begin() as conn:
                    for data in chunk_data:
                        await conn.execute(text("""
                            INSERT OR REPLACE INTO chunks 
                            (chunk_id, doc_id, content, chunk_index, page_number, start_char, 
                             end_char, embedding_vector, metadata)
                            VALUES (:chunk_id, :doc_id, :content, :chunk_index, :page_number, 
                                   :start_char, :end_char, :embedding_vector, :metadata)
                        """), data)
            else:
                with self.engine.begin() as conn:
                    for data in chunk_data:
                        conn.execute(text("""
                            INSERT OR REPLACE INTO chunks 
                            (chunk_id, doc_id, content, chunk_index, page_number, start_char, 
                             end_char, embedding_vector, metadata)
                            VALUES (:chunk_id, :doc_id, :content, :chunk_index, :page_number, 
                                   :start_char, :end_char, :embedding_vector, :metadata)
                        """), data)
            
            logger.info(f"Stored {len(chunks)} chunks")
            
        except Exception as e:
            logger.error(f"Error storing chunks: {str(e)}")
            raise
    
    async def log_query(self, query_result: QueryResult):
        """Log query for analytics"""
        try:
            query_id = hashlib.md5(
                f"{query_result.query}{datetime.now().isoformat()}".encode()
            ).hexdigest()
            
            query_data = {
                'query_id': query_id,
                'query_text': query_result.query,
                'domain': query_result.domain.value if query_result.domain else None,
                'query_type': query_result.query_type.value if query_result.query_type else None,
                'confidence_score': query_result.confidence_score,
                'processing_time': query_result.processing_time,
                'matched_chunks_count': len(query_result.matched_chunks),
                'timestamp': datetime.now()
            }
            
            if self.is_async:
                async with self.engine.begin() as conn:
                    await conn.execute(text("""
                        INSERT INTO queries 
                        (query_id, query_text, domain, query_type, confidence_score, 
                         processing_time, matched_chunks_count, timestamp)
                        VALUES (:query_id, :query_text, :domain, :query_type, :confidence_score, 
                               :processing_time, :matched_chunks_count, :timestamp)
                    """), query_data)
                    
                    # Store query-chunk relationships
                    for chunk in query_result.matched_chunks:
                        await conn.execute(text("""
                            INSERT INTO query_chunks (query_id, chunk_id, similarity_score)
                            VALUES (:query_id, :chunk_id, :similarity_score)
                        """), {
                            'query_id': query_id,
                            'chunk_id': chunk.get('chunk_id'),
                            'similarity_score': chunk.get('similarity_score', 0.0)
                        })
            else:
                with self.engine.begin() as conn:
                    conn.execute(text("""
                        INSERT INTO queries 
                        (query_id, query_text, domain, query_type, confidence_score, 
                         processing_time, matched_chunks_count, timestamp)
                        VALUES (:query_id, :query_text, :domain, :query_type, :confidence_score, 
                               :processing_time, :matched_chunks_count, :timestamp)
                    """), query_data)
                    
                    # Store query-chunk relationships
                    for chunk in query_result.matched_chunks:
                        conn.execute(text("""
                            INSERT INTO query_chunks (query_id, chunk_id, similarity_score)
                            VALUES (:query_id, :chunk_id, :similarity_score)
                        """), {
                            'query_id': query_id,
                            'chunk_id': chunk.get('chunk_id'),
                            'similarity_score': chunk.get('similarity_score', 0.0)
                        })
            
        except Exception as e:
            logger.error(f"Error logging query: {str(e)}")
            # Don't raise - query logging shouldn't break the main flow
    
    async def get_documents(
        self,
        domain: Optional[Domain] = None,
        doc_type: Optional[DocumentType] = None,
        limit: int = 100,
        offset: int = 0
    ) -> List[DocumentInfo]:
        """Get list of documents with optional filtering"""
        try:
            query = "SELECT * FROM documents WHERE 1=1"
            params = {}
            
            if domain:
                query += " AND domain = :domain"
                params['domain'] = domain.value
            
            if doc_type:
                query += " AND doc_type = :doc_type"
                params['doc_type'] = doc_type.value
            
            query += " ORDER BY upload_timestamp DESC LIMIT :limit OFFSET :offset"
            params['limit'] = limit
            params['offset'] = offset
            
            if self.is_async:
                async with self.engine.begin() as conn:
                    result = await conn.execute(text(query), params)
                    rows = result.fetchall()
            else:
                with self.engine.begin() as conn:
                    result = conn.execute(text(query), params)
                    rows = result.fetchall()
            
            documents = []
            for row in rows:
                # Get chunk count
                chunk_count = await self._get_chunk_count(row.doc_id)
                
                doc_info = DocumentInfo(
                    doc_id=row.doc_id,
                    filename=row.filename,
                    doc_type=DocumentType(row.doc_type),
                    domain=Domain(row.domain),
                    upload_timestamp=row.upload_timestamp,
                    file_size=row.file_size,
                    chunk_count=chunk_count,
                    word_count=row.word_count,
                    page_count=row.page_count
                )
                documents.append(doc_info)
            
            return documents
            
        except Exception as e:
            logger.error(f"Error getting documents: {str(e)}")
            return []
    
    async def _get_chunk_count(self, doc_id: str) -> int:
        """Get chunk count for a document"""
        try:
            if self.is_async:
                async with self.engine.begin() as conn:
                    result = await conn.execute(
                        text("SELECT COUNT(*) as count FROM chunks WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    row = result.fetchone()
            else:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        text("SELECT COUNT(*) as count FROM chunks WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    row = result.fetchone()
            
            return row.count if row else 0
            
        except Exception as e:
            logger.error(f"Error getting chunk count: {str(e)}")
            return 0
    
    async def get_document_by_id(self, doc_id: str) -> Optional[DocumentInfo]:
        """Get document by ID"""
        try:
            if self.is_async:
                async with self.engine.begin() as conn:
                    result = await conn.execute(
                        text("SELECT * FROM documents WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    row = result.fetchone()
            else:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        text("SELECT * FROM documents WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    row = result.fetchone()
            
            if not row:
                return None
            
            chunk_count = await self._get_chunk_count(doc_id)
            
            return DocumentInfo(
                doc_id=row.doc_id,
                filename=row.filename,
                doc_type=DocumentType(row.doc_type),
                domain=Domain(row.domain),
                upload_timestamp=row.upload_timestamp,
                file_size=row.file_size,
                chunk_count=chunk_count,
                word_count=row.word_count,
                page_count=row.page_count
            )
            
        except Exception as e:
            logger.error(f"Error getting document by ID: {str(e)}")
            return None
    
    async def delete_document(self, doc_id: str) -> bool:
        """Delete a document and its chunks"""
        try:
            if self.is_async:
                async with self.engine.begin() as conn:
                    # Delete chunks first (foreign key constraint)
                    await conn.execute(
                        text("DELETE FROM chunks WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    
                    # Delete document
                    result = await conn.execute(
                        text("DELETE FROM documents WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    
                    return result.rowcount > 0
            else:
                with self.engine.begin() as conn:
                    # Delete chunks first
                    conn.execute(
                        text("DELETE FROM chunks WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    
                    # Delete document
                    result = conn.execute(
                        text("DELETE FROM documents WHERE doc_id = :doc_id"),
                        {'doc_id': doc_id}
                    )
                    
                    return result.rowcount > 0
            
        except Exception as e:
            logger.error(f"Error deleting document: {str(e)}")
            return False
    
    async def get_query_analytics(
        self,
        domain: Optional[Domain] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: int = 100,
        offset: int = 0
    ) -> QueryHistory:
        """Get query analytics and history"""
        try:
            # Build query with filters
            query = "SELECT * FROM queries WHERE 1=1"
            params = {}
            
            if domain:
                query += " AND domain = :domain"
                params['domain'] = domain.value
            
            if start_date:
                query += " AND timestamp >= :start_date"
                params['start_date'] = start_date
            
            if end_date:
                query += " AND timestamp <= :end_date"
                params['end_date'] = end_date
            
            query += " ORDER BY timestamp DESC LIMIT :limit OFFSET :offset"
            params['limit'] = limit
            params['offset'] = offset
            
            # Get queries
            if self.is_async:
                async with self.engine.begin() as conn:
                    result = await conn.execute(text(query), params)
                    rows = result.fetchall()
            else:
                with self.engine.begin() as conn:
                    result = conn.execute(text(query), params)
                    rows = result.fetchall()
            
            # Convert to QueryAnalytics objects
            queries = []
            total_confidence = 0
            total_processing_time = 0
            domain_counts = {}
            
            for row in rows:
                query_analytics = QueryAnalytics(
                    query_id=row.query_id,
                    query_text=row.query_text,
                    domain=Domain(row.domain) if row.domain else None,
                    query_type=row.query_type,
                    confidence_score=row.confidence_score or 0,
                    processing_time=row.processing_time or 0,
                    chunks_retrieved=row.matched_chunks_count or 0,
                    timestamp=row.timestamp,
                    user_feedback=row.user_feedback
                )
                queries.append(query_analytics)
                
                # Accumulate stats
                total_confidence += query_analytics.confidence_score
                total_processing_time += query_analytics.processing_time
                
                domain_key = query_analytics.domain.value if query_analytics.domain else 'unknown'
                domain_counts[domain_key] = domain_counts.get(domain_key, 0) + 1
            
            # Calculate averages
            count = len(queries)
            avg_confidence = total_confidence / count if count > 0 else 0
            avg_processing_time = total_processing_time / count if count > 0 else 0
            
            return QueryHistory(
                queries=queries,
                total_count=count,
                average_confidence=avg_confidence,
                average_processing_time=avg_processing_time,
                domain_distribution=domain_counts
            )
            
        except Exception as e:
            logger.error(f"Error getting query analytics: {str(e)}")
            return QueryHistory(
                queries=[],
                total_count=0,
                average_confidence=0,
                average_processing_time=0,
                domain_distribution={}
            )
    
    async def get_document_chunks(self, doc_id: str) -> List[Dict[str, Any]]:
        """Get all chunks for a document"""
        try:
            if self.is_async:
                async with self.engine.begin() as conn:
                    result = await conn.execute(
                        text("SELECT * FROM chunks WHERE doc_id = :doc_id ORDER BY chunk_index"),
                        {'doc_id': doc_id}
                    )
                    rows = result.fetchall()
            else:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        text("SELECT * FROM chunks WHERE doc_id = :doc_id ORDER BY chunk_index"),
                        {'doc_id': doc_id}
                    )
                    rows = result.fetchall()
            
            chunks = []
            for row in rows:
                chunk_data = {
                    'chunk_id': row.chunk_id,
                    'doc_id': row.doc_id,
                    'content': row.content,
                    'chunk_index': row.chunk_index,
                    'page_number': row.page_number,
                    'start_char': row.start_char,
                    'end_char': row.end_char,
                    'metadata': json.loads(row.metadata) if row.metadata else {},
                    'embedding': json.loads(row.embedding_vector) if row.embedding_vector else []
                }
                chunks.append(chunk_data)
            
            return chunks
            
        except Exception as e:
            logger.error(f"Error getting document chunks: {str(e)}")
            return []
    
    async def update_user_feedback(self, query_id: str, feedback: float):
        """Update user feedback for a query"""
        try:
            if not (1 <= feedback <= 5):
                raise ValueError("Feedback must be between 1 and 5")
            
            if self.is_async:
                async with self.engine.begin() as conn:
                    await conn.execute(
                        text("UPDATE queries SET user_feedback = :feedback WHERE query_id = :query_id"),
                        {'feedback': feedback, 'query_id': query_id}
                    )
            else:
                with self.engine.begin() as conn:
                    conn.execute(
                        text("UPDATE queries SET user_feedback = :feedback WHERE query_id = :query_id"),
                        {'feedback': feedback, 'query_id': query_id}
                    )
            
            logger.info(f"Updated user feedback for query {query_id}: {feedback}")
            
        except Exception as e:
            logger.error(f"Error updating user feedback: {str(e)}")
            raise
    
    async def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics"""
        try:
            stats = {}
            
            # Document stats
            if self.is_async:
                async with self.engine.begin() as conn:
                    # Total documents
                    result = await conn.execute(text("SELECT COUNT(*) as count FROM documents"))
                    stats['total_documents'] = result.fetchone().count
                    
                    # Documents by domain
                    result = await conn.execute(text("""
                        SELECT domain, COUNT(*) as count 
                        FROM documents 
                        GROUP BY domain
                    """))
                    stats['documents_by_domain'] = {row.domain: row.count for row in result.fetchall()}
                    
                    # Total chunks
                    result = await conn.execute(text("SELECT COUNT(*) as count FROM chunks"))
                    stats['total_chunks'] = result.fetchone().count
                    
                    # Total queries
                    result = await conn.execute(text("SELECT COUNT(*) as count FROM queries"))
                    stats['total_queries'] = result.fetchone().count
                    
                    # Average confidence
                    result = await conn.execute(text("""
                        SELECT AVG(confidence_score) as avg_confidence 
                        FROM queries 
                        WHERE confidence_score IS NOT NULL
                    """))
                    avg_row = result.fetchone()
                    stats['average_confidence'] = float(avg_row.avg_confidence) if avg_row.avg_confidence else 0
                    
                    # Queries by domain
                    result = await conn.execute(text("""
                        SELECT domain, COUNT(*) as count 
                        FROM queries 
                        WHERE domain IS NOT NULL
                        GROUP BY domain
                    """))
                    stats['queries_by_domain'] = {row.domain: row.count for row in result.fetchall()}
            else:
                with self.engine.begin() as conn:
                    # Total documents
                    result = conn.execute(text("SELECT COUNT(*) as count FROM documents"))
                    stats['total_documents'] = result.fetchone().count
                    
                    # Documents by domain
                    result = conn.execute(text("""
                        SELECT domain, COUNT(*) as count 
                        FROM documents 
                        GROUP BY domain
                    """))
                    stats['documents_by_domain'] = {row.domain: row.count for row in result.fetchall()}
                    
                    # Total chunks
                    result = conn.execute(text("SELECT COUNT(*) as count FROM chunks"))
                    stats['total_chunks'] = result.fetchone().count
                    
                    # Total queries
                    result = conn.execute(text("SELECT COUNT(*) as count FROM queries"))
                    stats['total_queries'] = result.fetchone().count
                    
                    # Average confidence
                    result = conn.execute(text("""
                        SELECT AVG(confidence_score) as avg_confidence 
                        FROM queries 
                        WHERE confidence_score IS NOT NULL
                    """))
                    avg_row = result.fetchone()
                    stats['average_confidence'] = float(avg_row.avg_confidence) if avg_row.avg_confidence else 0
                    
                    # Queries by domain
                    result = conn.execute(text("""
                        SELECT domain, COUNT(*) as count 
                        FROM queries 
                        WHERE domain IS NOT NULL
                        GROUP BY domain
                    """))
                    stats['queries_by_domain'] = {row.domain: row.count for row in result.fetchall()}
            
            return stats
            
        except Exception as e:
            logger.error(f"Error getting database stats: {str(e)}")
            return {}
    
    async def cleanup_old_queries(self, days_to_keep: int = 30):
        """Clean up old query logs"""
        try:
            cutoff_date = datetime.now() - timedelta(days=days_to_keep)
            
            if self.is_async:
                async with self.engine.begin() as conn:
                    result = await conn.execute(
                        text("DELETE FROM queries WHERE timestamp < :cutoff_date"),
                        {'cutoff_date': cutoff_date}
                    )
                    deleted_count = result.rowcount
            else:
                with self.engine.begin() as conn:
                    result = conn.execute(
                        text("DELETE FROM queries WHERE timestamp < :cutoff_date"),
                        {'cutoff_date': cutoff_date}
                    )
                    deleted_count = result.rowcount
            
            logger.info(f"Cleaned up {deleted_count} old query records")
            return deleted_count
            
        except Exception as e:
            logger.error(f"Error cleaning up old queries: {str(e)}")
            return 0
    
    async def backup_database(self, backup_path: str):
        """Create a backup of the database"""
        try:
            if "sqlite" in self.db_url:
                # For SQLite, copy the file
                import shutil
                db_file = self.db_url.replace("sqlite:///", "")
                shutil.copy2(db_file, backup_path)
                logger.info(f"SQLite database backed up to {backup_path}")
            else:
                # For PostgreSQL, use pg_dump (requires external tool)
                logger.warning("PostgreSQL backup requires external pg_dump tool")
                
        except Exception as e:
            logger.error(f"Error backing up database: {str(e)}")
            raise
    
    async def close(self):
        """Close database connections"""
        try:
            if self.is_async:
                await self.engine.dispose()
            else:
                self.engine.dispose()
            logger.info("Database connections closed")
        except Exception as e:
            logger.error(f"Error closing database: {str(e)}")
