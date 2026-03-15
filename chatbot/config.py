import os
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

# --- Ollama LLM (local dev) ---
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "phi3.5")

# --- Groq LLM (deployed / default) ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")

# --- LLM shared params ---
LLM_TEMPERATURE = float(os.getenv("LLM_TEMPERATURE", "0.2"))
LLM_MAX_TOKENS = int(os.getenv("LLM_MAX_TOKENS", "400"))
LLM_TOP_P = float(os.getenv("LLM_TOP_P", "0.9"))

# --- Retrieval ---
RETRIEVAL_TOP_K = int(os.getenv("RETRIEVAL_TOP_K", "6"))
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "6000"))
MAX_CHUNK_CHARS = int(os.getenv("MAX_CHUNK_CHARS", "800"))
EXCLUDED_SECTIONS = {"References", "Application"}

# --- Conversation ---
MAX_HISTORY_TURNS = 3

# --- Embedding ---
EMBEDDING_SCRIPTS_DIR = PROJECT_ROOT / "embedding" / "scripts"
