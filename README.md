# 🧠 Mini Semantic Search API

This project implements a simple semantic search system using FastAPI and FAISS.

## 🚀 Features
- Search images or captions semantically using text or image query
- Uses CLIP from sentence-transformers for embeddings
- Stores embeddings in local FAISS index

## 📦 Setup
```bash
pip install -r requirements.txt
python app/data_loader.py
uvicorn app.main:app --reload
```

## 🔍 Endpoints
- `POST /search-text`: `{ "query": "a man on a horse" }`
- `POST /search-image`: `multipart/form-data` with image file

## 🐳 Docker
```bash
docker build -t semantic-search .
docker run -p 8000:8000 semantic-search
```

## ✅ Example
```bash
curl -X POST http://localhost:8000/search-text -H "Content-Type: application/json" -d '{"query": "a man riding a horse"}'
```

## 🧪 Tests
```bash
pytest tests/
```