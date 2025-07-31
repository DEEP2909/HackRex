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
@router.post("/query/detailed", response_model=DetailedQueryResponse)
async def process_detailed_query(
    request: DetailedQueryRequest,
    components = Depends(get_components)
) -> DetailedQueryResponse:
    """Process a single query with detailed response"""
    try:
        start_time = time.time()
        embedding_engine = components['embedding_engine']
        llm_processor = components['llm_processor']
        clause_matcher = components['clause_matcher']
        db_manager = components['db_manager']
        
        # Search for relevant chunks
        relevant_chunks = await embedding_engine.search_similar_chunks(
            request.query, 
            top_k=request.max_chunks,
            similarity_threshold=request.similarity_threshold
        )
        
        # Filter by documents if specified
        if request.documents:
            relevant_chunks = [
                (chunk, score) for chunk, score in relevant_chunks
                if chunk.get('doc_id') in request.documents
            ]
        
        # Determine domain
        domain = request.domain or _detect_domain(request.query)
        
        # Process with LLM
        result = await llm_processor.process_query(
            request.query, relevant_chunks, domain.value, temperature=request.temperature
        )
        
        # Log query
        await db_manager.log_query(result)
        
        # Convert chunks to ChunkInfo
        from ..models.document import ChunkInfo
        chunk_infos = [
            ChunkInfo(
                chunk_id=chunk.get('chunk_id', ''),
                doc_id=chunk.get('doc_id', ''),
                content=chunk.get('content', ''),
                chunk_index=chunk.get('chunk_index', 0),
                similarity_score=score,
                metadata=chunk.get('metadata', {})
            )
            for chunk, score in relevant_chunks
        ]
        
        processing_time = time.time() - start_time
        
        return DetailedQueryResponse(
            query=request.query,
            answer=result.structured_response.get('answer', 'No answer found'),
            confidence_score=result.confidence_score,
            processing_time=processing_time,
            matched_chunks=chunk_infos,
            decision_rationale=result.decision_rationale if request.include_rationale else None,
            query_type=request.query_type,
            domain=domain,
            metadata={
                'chunks_count': len(chunk_infos),
                'average_similarity': sum(score for _, score in relevant_chunks) / len(relevant_chunks) if relevant_chunks else 0
            }
        )
        
    except Exception as e:
        logger.error(f"Error in detailed query: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/query/bulk", response_model=BulkQueryResponse)
async def process_bulk_queries(
    request: BulkQueryRequest,
    components = Depends(get_components)
) -> BulkQueryResponse:
    """Process multiple queries in bulk"""
    try:
        start_time = time.time()
        
        if request.parallel_processing:
            # Process queries in parallel with concurrency limit
            semaphore = asyncio.Semaphore(request.max_concurrent)
            
            async def process_single_query(query_req):
                async with semaphore:
                    return await process_detailed_query(query_req, components)
            
            tasks = [process_single_query(query) for query in request.queries]
            results = await asyncio.gather(*tasks, return_exceptions=True)
        else:
            # Process queries sequentially
            results = []
            for query_req in request.queries:
                try:
                    result = await process_detailed_query(query_req, components)
                    results.append(result)
                except Exception as e:
                    logger.error(f"Error processing query: {str(e)}")
                    results.append(e)
        
        # Separate successful and failed results
        successful_results = [r for r in results if isinstance(r, DetailedQueryResponse)]
        failed_count = len(results) - len(successful_results)
        
        processing_time = time.time() - start_time
        
        return BulkQueryResponse(
            results=successful_results,
            total_queries=len(request.queries),
            successful_queries=len(successful_results),
            failed_queries=failed_count,
            total_processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error in bulk query processing: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents", response_model=List[DocumentInfo])
async def list_documents(
    domain: Optional[Domain] = None,
    doc_type: Optional[DocumentType] = None,
    limit: int = 100,
    offset: int = 0,
    components = Depends(get_components)
) -> List[DocumentInfo]:
    """List uploaded documents"""
    try:
        db_manager = components['db_manager']
        documents = await db_manager.get_documents(
            domain=domain, doc_type=doc_type, limit=limit, offset=offset
        )
        return documents
    except Exception as e:
        logger.error(f"Error listing documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/documents/{doc_id}", response_model=DocumentInfo)
async def get_document(
    doc_id: str,
    components = Depends(get_components)
) -> DocumentInfo:
    """Get document information"""
    try:
        db_manager = components['db_manager']
        document = await db_manager.get_document_by_id(doc_id)
        if not document:
            raise HTTPException(status_code=404, detail="Document not found")
        return document
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.delete("/documents/{doc_id}")
async def delete_document(
    doc_id: str,
    components = Depends(get_components)
):
    """Delete a document and its chunks"""
    try:
        db_manager = components['db_manager']
        embedding_engine = components['embedding_engine']
        
        # Delete from database
        deleted = await db_manager.delete_document(doc_id)
        if not deleted:
            raise HTTPException(status_code=404, detail="Document not found")
        
        # Remove from vector index
        await embedding_engine.remove_document_chunks(doc_id)
        
        return {"message": f"Document {doc_id} deleted successfully"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error deleting document: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.post("/documents/search", response_model=DocumentSearchResponse)
async def search_documents(
    request: DocumentSearchRequest,
    components = Depends(get_components)
) -> DocumentSearchResponse:
    """Search documents by content"""
    try:
        start_time = time.time()
        embedding_engine = components['embedding_engine']
        
        # Search for similar chunks
        relevant_chunks = await embedding_engine.search_similar_chunks(
            request.query,
            top_k=request.max_results,
            similarity_threshold=request.similarity_threshold,
            domain_filter=request.domain,
            doc_type_filter=request.doc_types
        )
        
        # Convert to ChunkInfo
        from ..models.document import ChunkInfo
        chunk_infos = [
            ChunkInfo(
                chunk_id=chunk.get('chunk_id', ''),
                doc_id=chunk.get('doc_id', ''),
                content=chunk.get('content', ''),
                chunk_index=chunk.get('chunk_index', 0),
                similarity_score=score,
                metadata=chunk.get('metadata', {})
            )
            for chunk, score in relevant_chunks
        ]
        
        processing_time = time.time() - start_time
        
        return DocumentSearchResponse(
            query=request.query,
            results=chunk_infos,
            total_found=len(chunk_infos),
            processing_time=processing_time
        )
        
    except Exception as e:
        logger.error(f"Error searching documents: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@router.get("/analytics/queries", response_model=QueryHistory)
async def get_query_analytics(
    domain: Optional[Domain] = None,
    limit: int = 100,
    offset: int = 0,
    components = Depends(get_components)
) -> QueryHistory:
    """Get query analytics and history"""
    try:
        db_manager = components['db_manager']
        analytics = await db_manager.get_query_analytics(
            domain=domain, limit=limit, offset=offset
        )
        return analytics
    except Exception as e:
        logger.error(f"Error getting query analytics: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# Helper functions
def _detect_domain(query: str) -> Domain:
    """Detect domain from query content"""
    query_lower = query.lower()
    
    domain_keywords = {
        Domain.INSURANCE: ['policy', 'coverage', 'premium', 'deductible', 'claim', 'benefit'],
        Domain.LEGAL: ['contract', 'agreement', 'liability', 'clause', 'terms', 'conditions'],
        Domain.HR: ['employee', 'benefits', 'salary', 'performance', 'leave', 'policy'],
        Domain.COMPLIANCE: ['regulation', 'audit', 'compliance', 'requirement', 'standard']
    }
    
    domain_scores = {}
    for domain, keywords in domain_keywords.items():
        score = sum(1 for keyword in keywords if keyword in query_lower)
        domain_scores[domain] = score
    
    return max(domain_scores, key=domain_scores.get) if any(domain_scores.values()) else Domain.GENERAL

async def _process_document_background(
    file_path: str, 
    doc_metadata, 
    document_processor, 
    embedding_engine, 
    db_manager
):
    """Background task for document processing"""
    try:
        import os
        from ..models.document import ProcessedChunk
        
        # Process document
        text_chunks = await document_processor.process_document(file_path, doc_metadata)
        
        # Create processed chunks
        processed_chunks = []
        for i, chunk_text in enumerate(text_chunks):
            chunk = ProcessedChunk(
                chunk_id=f"{doc_metadata.doc_id}_{i}",
                doc_id=doc_metadata.doc_id,
                content=chunk_text,
                embedding=[],  # Will be filled by embedding engine
                chunk_index=i,
                metadata={
                    'domain': doc_metadata.domain.value,
                    'filename': doc_metadata.filename,
                    'doc_type': doc_metadata.doc_type.value
                }
            )
            processed_chunks.append(chunk)
        
        # Index chunks
        await embedding_engine.index_chunks(processed_chunks)
        
        # Store in database
        await db_manager.store_document(doc_metadata)
        await db_manager.store_chunks(processed_chunks)
        
        # Clean up temp file
        os.remove(file_path)
        
        logger.info(f"Successfully processed document {doc_metadata.filename} with {len(processed_chunks)} chunks")
        
    except Exception as e:
        logger.error(f"Error in background document processing: {str(e)}")
        # Clean up temp file on error
        try:
            os.remove(file_path)
        except:
            pass
