import nltk
import json
from nltk.tokenize import word_tokenize

nltk.download("punkt")
nltk.download("punkt_tab") 

CHUNK_SIZE = 300
OVERLAP = 50

# Function to chunk text into overlapping segments
def chunk_text(text):
    words = text.split() 
    chunks = []
    start = 0
    step = CHUNK_SIZE - OVERLAP

    # Create chunks with overlap
    while start + CHUNK_SIZE <= len(words):
        chunk = words[start:start + CHUNK_SIZE]
        chunks.append(" ".join(chunk))
        start += step

    return chunks

if __name__ == "__main__":
    # Load raw corpus
    with open("data/raw_corpus.json") as f:
        corpus = json.load(f)

    chunked = []
    cid = 0

    # Chunk each document and assign chunk IDs
    for doc in corpus["documents"]:
        chunks = chunk_text(doc["text"])
        for c in chunks:
            chunked.append({
                "chunk_id": cid,
                "url": doc["url"],
                "title": doc["title"],
                "text": c
            })
            cid += 1

    # Save chunked corpus
    with open("data/corpus_chunks.json", "w") as f:
        json.dump(chunked, f, indent=2)
