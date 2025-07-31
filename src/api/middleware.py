"""
Middleware configuration for the FastAPI application.
"""
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from ..utils.logger import get_logger

logger = get_logger(__name__)

def setup_middleware(app: FastAPI):
    """
    Configures and adds middleware to the FastAPI application.
    """
    # CORS Middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # In production, restrict this to your frontend's domain
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Request Logging Middleware
    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        start_time = time.time()
        
        response = await call_next(request)
        
        process_time = time.time() - start_time
        response.headers["X-Process-Time"] = str(process_time)
        
        logger.info(
            f"request_path={request.url.path} "
            f"method={request.method} "
            f"status_code={response.status_code} "
            f"duration={process_time:.4f}s"
        )
        return response

    logger.info("Middleware configured: CORS and request logging enabled.")
