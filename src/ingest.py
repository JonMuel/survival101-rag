"""
INGEST-PIPELINE  (offline, einmalig ausfuehren)
================================================

Das ist die "Kueche" des RAG-Systems. Sie laeuft NICHT bei jeder Nutzerfrage,
sondern nur dann, wenn sich deine Dokumente aendern. Aufruf:

    python -m src.ingest

Die vier klassischen RAG-Schritte der Ingestion:

    1. LADEN     - Dokumente aus data/raw einlesen (PDF, Markdown, HTML ...)
    2. CHUNKING  - lange Texte in verdauliche Stuecke zerschneiden
    3. EMBEDDING - jeden Chunk in einen Vektor (Zahlenliste) uebersetzen
    4. SPEICHERN - Vektoren + Text in der Chroma-Datenbank ablegen

Warum getrennt von der App? Weil Embedding rechenintensiv ist. Wir wollen es
einmal erledigen und den fertigen Index dann nur noch laden - nicht bei jedem
Start neu berechnen. Genau diese Trennung brauchst du spaeter fuers Deployment.
"""

import chromadb
from llama_index.core import SimpleDirectoryReader, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.vector_stores.chroma import ChromaVectorStore

from src import config


def build_index() -> None:
    # --- 1. LADEN --------------------------------------------------------
    # SimpleDirectoryReader erkennt das Dateiformat automatisch und waehlt
    # den passenden Loader (Markdown, PDF, HTML ...). Das ist der Bequemlich-
    # keitsgewinn des Frameworks: das Parsing pro Quellentyp nimmt es dir ab.
    print(f"[1/4] Lade Dokumente aus: {config.DATA_DIR}")
    documents = SimpleDirectoryReader(
        input_dir=str(config.DATA_DIR),
        recursive=True,
    ).load_data()
    print(f"      -> {len(documents)} Dokument(e) geladen.")

    # --- 2. CHUNKING -----------------------------------------------------
    # Der SentenceSplitter schneidet an Satzgrenzen (nicht mitten im Wort)
    # und respektiert CHUNK_SIZE / CHUNK_OVERLAP aus der config.
    splitter = SentenceSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
    )

    # --- 3. EMBEDDING (lokal, gratis) ------------------------------------
    print(f"[2/4] Lade Embedding-Modell: {config.EMBED_MODEL}")
    print("      (beim ersten Mal wird das Modell heruntergeladen - kann dauern)")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)

    # --- 4. SPEICHERN: Chroma vorbereiten --------------------------------
    # PersistentClient legt die Datenbank als Dateien in storage/ ab,
    # damit sie einen Programm-Neustart ueberlebt.
    print(f"[3/4] Oeffne Chroma-Datenbank in: {config.STORAGE_DIR}")
    chroma_client = chromadb.PersistentClient(path=str(config.STORAGE_DIR))
    # get_or_create: beim erneuten Ausfuehren nicht abstuerzen, sondern wieder-
    # verwenden. Fuer einen echten Neuaufbau die Collection vorher loeschen.
    chroma_collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
    storage_context = StorageContext.from_defaults(vector_store=vector_store)

    # from_documents fuehrt Chunking + Embedding + Speichern in einem Rutsch aus.
    print("[4/4] Chunke, embedde und schreibe in die Datenbank ...")
    VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context,
        embed_model=embed_model,
        transformations=[splitter],
        show_progress=True,
    )

    n = chroma_collection.count()
    print(f"\nFertig! {n} Chunks liegen jetzt im Index unter {config.STORAGE_DIR}")


if __name__ == "__main__":
    build_index()
