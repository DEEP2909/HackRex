#!/usr/bin/env python3
"""
Main entry point for the LLM-Powered Intelligent Query-Retrieval System
"""

import asyncio
import uvicorn
from fastapi import FastAPI
from contextlib import asynccontextmanager

from config.settings import Settings
from src.api.routes import router
from src.api.middleware import setup_middleware
from src.database.manager import DatabaseManager
from src.processors.embedding_engine import EmbeddingSearchEngine
from src.processors.llm_processor import LLMQueryProcessor
from src.processors.document_processor import DocumentProcessor
from src.processors.clause_matcher import ClauseMatchingEngine
from src.utils.logger import setup_logger

# Initialize logger
logger = setup_logger(__name__)

# Global components
embedding_engine = None
llm_processor = None
document_processor = None
clause_matcher = None
db_manager = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan management"""
    global embedding_engine, llm_processor, document_processor, clause_matcher, db_manager
    
    logger.info("Starting LLM Query Retrieval System...")
    
    try:
        # Initialize components
        settings = Settings()
        
        # Database
        db_manager = DatabaseManager(settings.database_url)
        await db_manager.initialize()
        
        # Document processor
        document_processor = DocumentProcessor()
        
        # Embedding engine
        embedding_engine = EmbeddingSearchEngine(settings.embedding_model)
        await embedding_engine.initialize()
        
        # LLM processor
        llm_processor = LLMQueryProcessor(settings.llm_model)
        
        # Clause matcher
        clause_matcher = ClauseMatchingEngine(embedding_engine)
        
        # Store components in app state
        app.state.db_manager = db_manager
        app.state.document_processor = document_processor
        app.state.embedding_engine = embedding_engine
        app.state.llm_processor = llm_processor
        app.state.clause_matcher = clause_matcher
        
        logger.info("System initialization completed successfully")
        
        yield
        
    except Exception as e:
        logger.error(f"Failed to initialize system: {str(e)}")
        raise
    finally:
        logger.info("Shutting down system...")
        if embedding_engine:
            await embedding_engine.cleanup()

def create_app() -> FastAPI:
    """Create and configure FastAPI application"""
    app = FastAPI(
        title="LLM-Powered Query Retrieval System",
        description="Intelligent document processing and query retrieval system",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Setup middleware
    setup_middleware(app)
    
    # Include routes
    app.include_router(router, prefix="/api/v1")
    
    return app

def main():
    """Main function to run the application"""
    settings = Settings()
    app = create_app()
    
    logger.info(f"Starting server on {settings.host}:{settings.port}")
    
    uvicorn.run(
        app,
        host=settings.host,
        port=settings.port,
        log_level=settings.log_level.lower(),
        reload=settings.debug
    )

if __name__ == "__main__":
    main()
