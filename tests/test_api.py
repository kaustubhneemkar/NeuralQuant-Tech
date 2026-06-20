import sys
from pathlib import Path
from unittest.mock import MagicMock

# Mock the google.genai module BEFORE any project imports
# This prevents ImportError in CI where the full SDK may not install cleanly
genai_mock = MagicMock()
google_mock = MagicMock()
google_mock.genai = genai_mock
sys.modules.setdefault("google.genai", genai_mock)
sys.modules.setdefault("google", google_mock)

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app
from fastapi.testclient import TestClient

client = TestClient(app)

def test_health_endpoint():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert "model" in response.json()

def test_tools_endpoint():
    response = client.get("/tools")
    assert response.status_code == 200
    assert "tools" in response.json()
    tools = response.json()["tools"]
    assert len(tools) == 6
    assert any(t["name"] == "get_market_data" for t in tools)

def test_root_serves_ui():
    response = client.get("/")
    assert response.status_code == 200
    # Should serve the index.html page
    assert "ALPHA-LENS" in response.text
