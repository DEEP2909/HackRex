"""
API Routes for the LLM Query Retrieval System
"""
import asyncio
import hashlib
import time
import os
from datetime import datetime
from typing import List, Optional
from fastapi import APIRouter, HTTPException, UploadFile, File, Depends, Request, BackgroundTasks
from ..models.query import (
    QueryRequest, QueryResponse, DetailedQueryRequest, DetailedQueryResponse,
    BulkQueryRequest, BulkQueryResponse, QueryHistory
)
from ..models.document import (
    DocumentUploadRequest, DocumentUploadResponse, DocumentInfo,
    DocumentSearchRequest, DocumentSearchResponse, Domain, DocumentType
)
from ..utils.logger import get_logger
from ..utils.helpers import download_file_from_url
from ..processors.document_processor import DocumentProcessor
from ..processors.embedding_engine import EmbeddingSearchEngine
from ..processors.llm_processor import LLMQueryProcessor
from ..processors.clause_matcher import ClauseMatchingEngine
from ..database.manager import DatabaseManager

logger = get_logger(__name__)
router = APIRouter()

# Dependency injection
def get_components(request: Request):
    """Get system components from app state"""
    return {
        'db_manager': request.app.state.db_manager,
        'document_processor': request.app.state.document_processor,
        'embedding_engine': request.app.state.embedding_engine,
        'llm_processor': request.app.state.llm_processor,
        'clause_matcher': request.app.state.clause_matcher
    }

@router.post("/hackrx/run", response_model=QueryResponse)
async def process_queries(
    request: QueryRequest,
    components=Depends(get_components)
) -> QueryResponse:
    """
    Main endpoint for processing queries against documents.
    This matches the exact API specification from the problem statement.
    It downloads, processes, and queries documents on-the-fly for the request.
    """
    start_time = time.time()
    document_processor: DocumentProcessor = components['document_processor']
    embedding_engine: EmbeddingSearchEngine = components['embedding_engine']
    llm_processor: LLMQueryProcessor = components['llm_processor']
    
    processed_doc_ids = []
    temp_files_to_clean = []

    try:
        # Step 1 & 2: Download and Process each document from the URLs
        for doc_url in request.documents:
            try:
                # Download document content
                file_content = await download_file_from_url(doc_url)
                doc_id = hashlib.md5(file_content).hexdigest()
                filename = os.path.basename(doc_url.split('?')[0])
                file_ext = filename.split('.')[-1].lower()

                # Save file temporarily for processing
                temp_path = f"./data/temp/{doc_id}_{filename}"
                with open(temp_path, "wb") as f:
                    f.write(file_content)
                temp_files_to_clean.append(temp_path)

                # Create metadata
                metadata = DocumentMetadata(
                    doc_id=doc_id,
                    filename=filename,
                    doc_type=DocumentType(file_ext),
                    upload_timestamp=datetime.now(),
                    file_size=len(file_content),
                    content_hash=doc_id,
                    domain=request.domain or Domain.GENERAL
                )

                # Process document into chunks
                text_chunks = await document_processor.process_document(temp_path, metadata)
                
                # Create ProcessedChunk objects
                from ..models.document import ProcessedChunk
                processed_chunks = []
                for i, chunk_text in enumerate(text_chunks):
                    chunk = ProcessedChunk(
                        chunk_id=f"{metadata.doc_id}_{i}",
                        doc_id=metadata.doc_id,
                        content=chunk_text,
                        embedding=[],
                        chunk_index=i,
                        metadata={'domain': metadata.domain.value, 'filename': metadata.filename}
                    )
                    processed_chunks.append(chunk)

                # Step 3: Index chunks in the embedding engine for this request
                if processed_chunks:
                    await embedding_engine.index_chunks(processed_chunks)
                    processed_doc_ids.append(doc_id)
                    logger.info(f"Temporarily indexed {len(processed_chunks)} chunks for doc_id: {doc_id}")

            except Exception as e:
                logger.error(f"Failed to process document from URL {doc_url}: {e}")
                # We can choose to continue or fail the whole request
                raise HTTPException(status_code=400, detail=f"Could not process document: {doc_url}")

        if not processed_doc_ids:
            raise HTTPException(status_code=400, detail="No documents could be processed.")

        # Step 4, 5, 6: Process each question against the newly indexed documents
        answers = []
        for question in request.questions:
            # Search for relevant chunks ONLY within the processed documents
            relevant_chunks = await embedding_engine.search_similar_chunks(
                question, 
                top_k=request.max_chunks,
                doc_ids=processed_doc_ids # CRITICAL: Filter by this request's documents
            )
            
            # Process with LLM
            result = await llm_processor.process_query(
                question, relevant_chunks, (request.domain or Domain.GENERAL).value
            )
            
            answers.append(result.structured_response.get('answer', 'No answer found.'))
        
        processing_time = time.time() - start_time
        logger.info(f"Completed processing {len(request.questions)} queries in {processing_time:.2f}s")
        
        return QueryResponse(answers=answers)
        
    except Exception as e:
        logger.error(f"Error in process_queries: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Query processing failed: {str(e)}")
    
    finally:
        # Cleanup: Remove temp files and optionally de-index chunks
        for f in temp_files_to_clean:
            if os.path.exists(f):
                os.remove(f)
        # For a truly isolated system, you would also remove the chunks from the index
        # For simplicity here, we assume the index can be transient or cleaned up later.
        for doc_id in processed_doc_ids:
            await embedding_engine.remove_document_chunks(doc_id)
        logger.info("Cleaned up temporary files and indexed chunks for request.")

# (Keep all other existing routes like /upload-document, /health, etc., as they are)
# ... rest of the original routes.py file ...
@router.post("/upload-document", response_model=DocumentUploadResponse)
async def upload_document(
    file: UploadFile = File(...),
    domain: Domain = Domain.GENERAL,
    language: str = "en",
    background_tasks: BackgroundTasks = BackgroundTasks(),
    components = Depends(get_components)
) -> DocumentUploadResponse:
    """Upload and process a document"""
    try:
        start_time = time.time()
        document_processor = components['document_processor']
        embedding_engine = components['embedding_engine']
        db_manager = components['db_manager']
        
        # Validate file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        file_ext = "." + file.filename.split(".")[-1].lower()
        if file_ext not in [".pdf", ".docx", ".txt", ".html", ".eml"]:
            raise HTTPException(status_code=400, detail=f"Unsupported file type: {file_ext}")
        
        # Read file content
        content = await file.read()
        if len(content) > 10 * 1024 * 1024:  # 10MB limit
            raise HTTPException(status_code=400, detail="File too large (max 10MB)")
        
        # Create document metadata
        from ..models.document import DocumentMetadata, DocumentType
        
        doc_metadata = DocumentMetadata(
            doc_id=hashlib.md5(content).hexdigest(),
            filename=file.filename,
            doc_type=DocumentType(file_ext[1:]),  # Remove the dot
            upload_timestamp=datetime.now(),
            file_size=len(content),
            content_hash=hashlib.md5(content).hexdigest(),
            domain=domain,
            language=language
        )
        
        # Save file temporarily and process
        temp_path = f"./data/temp/{doc_metadata.doc_id}_{file.filename}"
        with open(temp_path, "wb") as f:
            f.write(content)
        
        # Process document in background
        background_tasks.add_task(
            _process_document_background,
            temp_path, doc_metadata, document_processor, embedding_engine, db_manager
        )
        
        processing_time = time.time() - start_time
        
        return DocumentUploadResponse(
            doc_id=doc_metadata.doc_id,
            filename=file.filename,
            message=f"Document {file.filename} uploaded and processing started",
            chunks_count=0,  # Will be updated after processing
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error uploading document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.now(),
        "version": "1.0.0"
    }
# ... and so on for the rest of the file's original functions.
