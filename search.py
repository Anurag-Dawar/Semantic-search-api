from app.embeddings import encode_text, encode_image
import faiss
import numpy as np
import json
import os

index = faiss.IndexFlatIP(512)
metadata = []


nlp = spacy.load("en_core_web_sm")

example_meta_data_set = {"kid","boy","infant","girl","baby"
                        "ball" ,"football","basketball","tennis ball"}



def extract_entities(text: str):
    doc = nlp(text)
    entities = set()

    # Named entities (e.g., people, locations)
    for ent in doc.ents:
        entities.add(ent.text.lower())

    # Nouns (e.g., boy, dog, park)
    for token in doc:
        if token.pos_ in ["NOUN", "PROPN"]:
            entities.add(token.lemma_.lower())

    return list(entities)

def stricter_match(query: str, candidate_caption: str) -> bool:
    required_phrases = extract_entities(query)
    caption_lower = candidate_caption.lower()
    return all(phrase in caption_lower for phrase in required_phrases)   
    



 

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


# Dummy function to simulate entity extraction from the query.
# Replace this with a real NLP-based implementation.
def extract_entities(query: str) -> set:
    # Example: extracting words by splitting; in practice use Named Entity Recognition, POS tagging, etc.
    return set(query.lower().split())


# Dummy keyword set for matching.
# In practice, this can be dynamically generated or based on domain knowledge.
keyword_set = {"ai", "robotics", "ml"}  


def semantic_match(query: str, candidate_caption: str) -> bool:
    """
    Checks if the candidate caption semantically matches the query
    based on whether all required keywords are present in the extracted entities.

    Parameters:
    - query (str): The input query string.
    - candidate_caption (str): The caption to compare against the query.

    Returns:
    - bool: True if there is a semantic match, False otherwise.
    """

    # Extract relevant entities or keywords from the query
    required_phrases = extract_entities(query)

    # Check if all required keywords are present in the extracted phrases
    has_keyword_overlap = all(word in required_phrases for word in keyword_set)

    return has_keyword_overlap


def filter_candidates(query: str, candidates: list) -> list:
    """
    Filters a list of candidate captions to only include those that semantically match the query.

    Parameters:
    - query (str): The input query string.
    - candidates (list): A list of candidate dicts, each containing a 'caption' key.

    Returns:
    - list: Filtered list of candidates with semantically matching captions.
    """

    return [
        c for c in candidates
        if semantic_match(query, c["caption"])
    ]

def search_by_text(query: str):
    vec = encode_text(query)
    return {"results": find_similar(vec)}

def search_by_image(image_bytes):
    vec = encode_image(image_bytes)
    return {"results": find_similar(vec)}
