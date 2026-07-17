"""
Central configuration.

All the "knobs" of the RAG system in one place. When you experiment while
learning (different embedding model, larger chunks, different LLM), you change
it here - not scattered across the code.
"""

from pathlib import Path

# --- Paths ---------------------------------------------------------------
# PROJECT_ROOT points at the survival101-rag folder (one level above /src).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"      # source documents live here
STORAGE_DIR = PROJECT_ROOT / "storage"        # ingest.py writes the index here

# Name of the Chroma collection (one database can hold multiple collections).
COLLECTION_NAME = "survival101"

# --- Embeddings ----------------------------------------------------------
# Small, multilingual model - runs locally on the CPU, costs nothing.
# On first run it is downloaded once (~120 MB).
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking ------------------------------------------------------------
# How large is a text piece (in tokens) and how much do pieces overlap?
# Overlap makes sure a thought is not unluckily cut in half at a chunk
# boundary. Good starting values to experiment with:
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Retrieval -----------------------------------------------------------
# How many of the most similar chunks do we fetch as context for the LLM?
TOP_K = 4

# --- LLM (Groq free tier) ------------------------------------------------
# Note: Groq model names change occasionally. If a call fails, check the
# current names at https://console.groq.com/docs/models
LLM_MODEL = "llama-3.3-70b-versatile"
