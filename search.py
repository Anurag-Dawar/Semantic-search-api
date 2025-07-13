from app.embeddings import encode_text, encode_image
import faiss
import numpy as np
import json
import os

index = faiss.IndexFlatIP(512)
metadata = []

def init_index():
    with open("data/captions.json") as f:
        captions = json.load(f)
    for filename, caption in captions.items():
        vec = encode_text(caption).numpy()
        norm = np.linalg.norm(vec)
        normed_vec = vec / norm
        index.add(np.array([normed_vec]))
        metadata.append({"id": filename, "caption": caption})

init_index()

def find_similar(vec, top_k=5):
    vec = vec.detach().numpy().astype("float32")
    vec = vec / np.linalg.norm(vec)
    D, I = index.search(np.array([vec]), top_k)
    return [
        {
            "id": metadata[i]["id"],
            "caption": metadata[i]["caption"],
            "score": float(D[0][idx])
        }
        for idx, i in enumerate(I[0]) if i < len(metadata)
    ]

def search_by_text(query: str):
    vec = encode_text(query)
    return {"results": find_similar(vec)}

def search_by_image(image_bytes):
    vec = encode_image(image_bytes)
    return {"results": find_similar(vec)}
