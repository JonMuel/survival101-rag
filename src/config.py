"""
Zentrale Konfiguration.

Alle "Stellschrauben" des RAG-Systems an einem Ort. Wenn du beim Lernen
experimentierst (anderes Embedding-Modell, groessere Chunks, anderes LLM),
aenderst du es hier - und nicht verstreut im Code.
"""

from pathlib import Path

# --- Pfade ---------------------------------------------------------------
# PROJECT_ROOT zeigt auf den survival101-rag-Ordner (eine Ebene ueber /src).
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "raw"      # hier liegen die Quelldokumente
STORAGE_DIR = PROJECT_ROOT / "storage"        # hierhin schreibt ingest.py den Index

# Name der Chroma-Collection (eine Datenbank kann mehrere Collections halten).
COLLECTION_NAME = "survival101"

# --- Embeddings ----------------------------------------------------------
# Kleines, mehrsprachiges Modell - laeuft lokal auf der CPU, kostet nichts.
# Beim ersten Start wird es einmalig heruntergeladen (~120 MB).
EMBED_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# --- Chunking ------------------------------------------------------------
# Wie gross ist ein Textstueck (in Tokens) und wie stark ueberlappen sie?
# Ueberlappung sorgt dafuer, dass ein Gedanke nicht ungluecklich an einer
# Chunk-Grenze zerrissen wird. Gute erste Werte zum Experimentieren:
CHUNK_SIZE = 512
CHUNK_OVERLAP = 64

# --- Retrieval -----------------------------------------------------------
# Wie viele der aehnlichsten Chunks holen wir als Kontext fuer das LLM?
TOP_K = 4

# --- LLM (Groq Free-Tier) ------------------------------------------------
# Hinweis: Modellnamen bei Groq aendern sich gelegentlich. Falls ein Aufruf
# fehlschlaegt, aktuelle Namen unter https://console.groq.com/docs/models pruefen.
LLM_MODEL = "llama-3.3-70b-versatile"
