import os
from dotenv import load_dotenv

base_dir = os.path.dirname(os.path.abspath(__file__))
env_path = os.path.join(base_dir, ".env")
load_dotenv(dotenv_path=env_path)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = "llama-3.3-70b-versatile"

EMBED_MODEL = "all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

CHUNK_SIZE = 500
CHUNK_OVERLAP = 50

FAISS_INDEX_PATH = "index/medbridge.faiss"
METADATA_PATH = "index/metadata.pkl"
CHUNKS_PATH = "data/processed/chunks.json"
RAW_DATA_DIR = "data/raw"
GUIDELINES_DIR = "data/raw/guidelines"

TOP_K_RETRIEVAL = 20
TOP_N_RERANK = 5

CONFIDENCE_THRESHOLDS = {"high": 0.7, "medium": 0.4}
