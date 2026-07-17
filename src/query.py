"""
RAG-KERN  (laeuft bei jeder Nutzerfrage)
========================================

Hier passiert das eigentliche "Retrieval-Augmented Generation". Ablauf, wenn
ein Nutzer eine Frage stellt:

    1. RETRIEVAL  - Frage embedden und die aehnlichsten Chunks aus Chroma holen
    2. AUGMENT    - diese Chunks als Kontext in einen Prompt packen
    3. GENERATION - das LLM (Groq) antwortet AUSSCHLIESSLICH auf Basis des Kontexts

Der letzte Punkt ist der Kern der Idee: Das LLM soll nicht frei aus seinem
Weltwissen fabulieren, sondern sich auf deine Quellen stuetzen. Das reduziert
Halluzinationen - besonders wichtig bei Survival-Themen, wo falsche Infos
gefaehrlich sein koennen.
"""

import os

import chromadb
from dotenv import load_dotenv
from llama_index.core import VectorStoreIndex
from llama_index.core.prompts import PromptTemplate
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.groq import Groq
from llama_index.vector_stores.chroma import ChromaVectorStore

from src import config

# .env laden, damit GROQ_API_KEY verfuegbar ist.
load_dotenv()

# System-Prompt: gibt dem LLM klare Leitplanken. Das "sag ehrlich, wenn du es
# nicht weisst" ist deine wichtigste Guardrail gegen erfundene Antworten.
QA_PROMPT = PromptTemplate(
    "Du bist 'Survival101', ein sachlicher Survival-Assistent.\n"
    "Beantworte die Frage AUSSCHLIESSLICH auf Basis des folgenden Kontexts.\n"
    "Wenn der Kontext die Antwort nicht hergibt, sage ehrlich, dass du es auf\n"
    "Grundlage der vorhandenen Quellen nicht beantworten kannst - erfinde nichts.\n"
    "Weise bei sicherheitskritischen Themen (Medizin, giftige Pflanzen) darauf\n"
    "hin, dass dies kein Ersatz fuer Fachwissen oder aerztliche Hilfe ist.\n\n"
    "--- KONTEXT ---\n{context_str}\n--- ENDE KONTEXT ---\n\n"
    "Frage: {query_str}\n"
    "Antwort (auf Deutsch):"
)


def load_query_engine():
    """Laedt den bereits gebauten Index und haengt das LLM an.

    Wird von der App genau einmal aufgerufen (und dort gecached), damit nicht
    bei jeder Frage das Modell neu geladen wird.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY fehlt. Lege eine .env-Datei an (Vorlage: .env.example)."
        )

    # Wichtig: dasselbe Embedding-Modell wie beim Ingest, sonst passen die
    # Vektoren nicht zusammen und das Retrieval liefert Unsinn.
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    llm = Groq(model=config.LLM_MODEL)

    # Bestehende Chroma-Collection oeffnen (NICHT neu bauen).
    chroma_client = chromadb.PersistentClient(path=str(config.STORAGE_DIR))
    chroma_collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Index aus dem vorhandenen Vektor-Store rekonstruieren.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # Der Query-Engine verbindet Retrieval (top_k Chunks) + Prompt + LLM.
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=config.TOP_K,
        text_qa_template=QA_PROMPT,
    )


if __name__ == "__main__":
    # Kleiner Selbsttest von der Kommandozeile: python -m src.query
    engine = load_query_engine()
    frage = "Wie mache ich Wasser aus einem Bach trinkbar?"
    print(f"Frage: {frage}\n")
    antwort = engine.query(frage)
    print(f"Antwort:\n{antwort}\n")
    print("Verwendete Quellen-Chunks:")
    for node in antwort.source_nodes:
        quelle = node.metadata.get("file_name", "unbekannt")
        print(f"  - {quelle} (Score: {node.score:.3f})")
