"""
INGEST PIPELINE  (offline, run once)
====================================

This is the "kitchen" of the RAG system. It does NOT run on every user question,
only when your documents change. Run it with:

    python -m src.ingest

The four classic RAG ingestion steps:

    1. LOAD      - read documents from data/raw (PDF, Markdown, HTML ...)
    2. CHUNK     - split long texts into digestible pieces
    3. EMBED     - translate each chunk into a vector (list of numbers)
    4. STORE     - save vectors + text in the Chroma database

Why separate from the app? Because embedding is compute-intensive. We want to do
it once and then only load the finished index - not recompute it on every start.
This exact separation is what you need later for deployment.
"""

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src import config


def build_index() -> None:
    # --- 1. LOAD ---------------------------------------------------------
    # SimpleDirectoryReader detects the file format automatically and picks
    # the matching loader (Markdown, PDF, HTML ...). This is the convenience
    # win of the framework: it handles per-source-type parsing for you.
    print(f"[1/4] Loading documents from: {config.DATA_DIR}")
    documents = SimpleDirectoryReader(
        input_dir=str(config.DATA_DIR),
        recursive=True,
    ).load_data()
    print(f"      -> {len(documents)} document(s) loaded.")

    # --- 2. CHUNK --------------------------------------------------------
    # The SentenceSplitter cuts at sentence boundaries (not mid-word) and
    # respects CHUNK_SIZE / CHUNK_OVERLAP from the config.
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # --- 3. EMBED (local, free) ------------------------------------------
    print(f"[2/4] Loading embedding model: {config.EMBED_MODEL}")
    print("      (the model is downloaded on first run - this may take a while)")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)

    # --- 4. STORE: prepare Chroma ----------------------------------------
    # PersistentClient stores the database as files in storage/, so it
    # survives a program restart.
    print(f"[3/4] Opening Chroma database in: {config.STORAGE_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(config.STORAGE_DIR))
    # get_or_create: don't crash on re-run, reuse instead. For a true rebuild,
    # delete the collection beforehand.
    chroma_collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # from_documents runs chunking + embedding + storing in one go.
    print("[4/4] Chunking, embedding and writing to the database ...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    n = chroma_collection.count()
    print(f"\nDone! {n} chunks are now in the index at {config.STORAGE_DIR}")


if __name__ == "__main__":
    build_index()
