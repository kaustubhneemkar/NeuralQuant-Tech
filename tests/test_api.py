import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from api.main import app

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
