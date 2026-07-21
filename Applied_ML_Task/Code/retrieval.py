import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer, CrossEncoder
import config

_embedder = None
_reranker = None
_index = None
_metadata = None

def get_embedder():
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer(config.EMBED_MODEL)
    return _embedder

def get_reranker():
    global _reranker
    if _reranker is None:
        _reranker = CrossEncoder(config.RERANK_MODEL)
    return _reranker

def load_index():
    global _index, _metadata
    if _index is None:
        _index = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.METADATA_PATH, "rb") as f:
            _metadata = pickle.load(f)
    return _index, _metadata

def search(query, top_k=config.TOP_K_RETRIEVAL):
    index, metadata = load_index()
    emb = get_embedder().encode([query], normalize_embeddings=True).astype("float32")
    scores, idxs = index.search(emb, top_k)
    results = []
    for score, idx in zip(scores[0], idxs[0]):
        if idx < 0:
            continue
        entry = dict(metadata[idx])
        entry["retrieval_score"] = float(score)
        results.append(entry)
    return results

def rerank(query, candidates, top_n=config.TOP_N_RERANK):
    if not candidates:
        return []
    reranker = get_reranker()
    pairs = [(query, c["text"]) for c in candidates]
    scores = reranker.predict(pairs)
    for i, s in enumerate(scores):
        candidates[i]["rerank_score"] = float(s)
    ranked = sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
    return ranked[:top_n]

def retrieve(query):
    candidates = search(query)
    return rerank(query, candidates)
