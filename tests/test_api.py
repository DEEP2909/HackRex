import pytest
import asyncio
from httpx import AsyncClient
from fastapi.testclient import TestClient

# It's better to create a fixture to initialize the app for tests
# This code assumes your main app can be imported and created for testing
from main import create_app

# Use the TestClient for synchronous tests or httpx for async against a running server
client = TestClient(create_app())

@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for our test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()

def test_health_check():
    """Tests the /health endpoint to ensure the server is running."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    assert response.json()["status"] == "healthy"

@pytest.mark.asyncio
async def test_process_queries_endpoint():
    """
    Tests the main /hackrx/run endpoint.
    NOTE: This is an integration test. It makes a live call to a PDF URL
    and expects the LLM API keys to be valid.
    """
    # Using AsyncClient to test an async endpoint
    async with AsyncClient(app=create_app(), base_url="http://test") as ac:
        request_body = {
            "documents": [
                # This is a publicly accessible sample PDF for testing
                "https://www.w3.org/WAI/ER/tests/xhtml/testfiles/resources/pdf/dummy.pdf"
            ],
            "questions": [
                "What is the content of this document?"
            ],
            "domain": "general",
            "max_chunks": 5
        }
        response = await ac.post("/api/v1/hackrx/run", json=request_body, timeout=60)

    # Assertions
    assert response.status_code == 200
    response_json = response.json()
    assert "answers" in response_json
    assert isinstance(response_json["answers"], list)
    assert len(response_json["answers"]) == 1
    # Check that the answer is a non-empty string
    assert isinstance(response_json["answers"][0], str)
    assert len(response_json["answers"][0]) > 0
