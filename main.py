from fastapi import FastAPI, UploadFile, File
from app.models import TextQuery
from app.search import search_by_text, search_by_image

app = FastAPI()

@app.post("/search-text")
def search_text_endpoint(query: TextQuery):
    return search_by_text(query.query)

@app.post("/search-image")
def search_image_endpoint(file: UploadFile = File(...)):
    image_bytes = file.file.read()
    return search_by_image(image_bytes)
