"""
Utility helper functions for the application.
"""
import httpx
import asyncio
from .logger import get_logger

logger = get_logger(__name__)

async def download_file_from_url(url: str, timeout: int = 30) -> bytes:
    """
    Asynchronously downloads a file from a given URL.

    Args:
        url (str): The URL of the file to download.
        timeout (int): Request timeout in seconds.

    Returns:
        bytes: The content of the file.

    Raises:
        Exception: If the download fails.
    """
    logger.info(f"Downloading file from URL: {url}")
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, follow_redirects=True, timeout=timeout)
            response.raise_for_status()  # Raises an exception for 4xx or 5xx status codes
            logger.info(f"Successfully downloaded file from {url} ({len(response.content)} bytes)")
            return response.content
        except httpx.HTTPStatusError as e:
            logger.error(f"HTTP error downloading {url}: {e}")
            raise Exception(f"Failed to download file: HTTP {e.response.status_code}")
        except Exception as e:
            logger.error(f"Error downloading {url}: {e}")
            raise Exception(f"An unexpected error occurred while downloading the file.")
