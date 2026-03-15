import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent.parent / ".env")

BASE_DIR = Path(__file__).resolve().parent.parent
PROJECT_ROOT = BASE_DIR.parent

# --- Paths ---
RAG_CHUNKS_PATH = PROJECT_ROOT / "scraper" / "data" / "processed" / "rag_chunks.json"
EMBEDDINGS_DIR = BASE_DIR / "data" / "embeddings"
EMBEDDINGS_FILE = EMBEDDINGS_DIR / "chunk_embeddings.npz"

# --- Embedding Model ---
EMBEDDING_MODEL_NAME = "abhinand/MedEmbed-large-v0.1"
EMBEDDING_DIM = 1024
BATCH_SIZE = 32
MAX_SEQ_LENGTH = 512

# --- Qdrant ---
QDRANT_URL = os.getenv("QDRANT_URL", None)
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY", None)
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", "6333"))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "uhc_policies")

# --- Search ---
TOP_K = 10

# --- Provider (for multi-provider extensibility) ---
PROVIDER_NAME = "UnitedHealthcare"
PROVIDER_SLUG = "uhc"
