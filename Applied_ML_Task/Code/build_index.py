import json, os, pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import config

def run():
    with open(config.CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"encoding {len(chunks)} chunks...")
    model = SentenceTransformer(config.EMBED_MODEL)
    texts = [c["text"] for c in chunks]
    embeddings = model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
    embeddings = np.array(embeddings).astype("float32")

    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    os.makedirs("index", exist_ok=True)
    faiss.write_index(index, config.FAISS_INDEX_PATH)
    with open(config.METADATA_PATH, "wb") as f:
        pickle.dump(chunks, f)

    print(f"index saved: {index.ntotal} vectors, dim={dim}")

if __name__ == "__main__":
    run()
