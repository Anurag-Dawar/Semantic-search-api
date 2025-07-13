from fastapi.testclient import TestClient
from app.main import app
import io

client = TestClient(app)

def test_search_text():
    response = client.post("/search-text", json={"query": "a sunset over mountains"})
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) > 0

def test_search_image():
    with open("data/images/image1.jpg", "rb") as f:
        response = client.post("/search-image", files={"file": ("image1.jpg", f, "image/jpeg")})
    assert response.status_code == 200
    assert "results" in response.json()
    assert len(response.json()["results"]) > 0
