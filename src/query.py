"""
RAG CORE  (runs on every user question)
=======================================

This is where the actual "Retrieval-Augmented Generation" happens. Flow when a
user asks a question:

    1. RETRIEVAL  - embed the question and fetch the most similar chunks from Chroma
    2. AUGMENT    - pack those chunks as context into a prompt
    3. GENERATION - the LLM (Groq) answers ONLY based on that context

The last point is the heart of the idea: the LLM should not freely invent from
its world knowledge but ground itself in your sources. This reduces
hallucinations - especially important for survival topics, where wrong
information can be dangerous.
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

# Load .env so GROQ_API_KEY is available.
load_dotenv()

# System prompt: gives the LLM clear guardrails. The "say honestly when you don't
# know" is your most important guardrail against invented answers.
QA_PROMPT = PromptTemplate(
    "You are 'Survival101', a factual survival assistant.\n"
    "Answer the question ONLY based on the following context.\n"
    "If the context does not contain the answer, honestly say that you cannot\n"
    "answer it based on the available sources - do not make anything up.\n"
    "For safety-critical topics (medicine, poisonous plants), point out that\n"
    "this is not a substitute for expert knowledge or medical help.\n\n"
    "--- CONTEXT ---\n{context_str}\n--- END CONTEXT ---\n\n"
    "Question: {query_str}\n"
    "Answer (in English):"
)


def load_query_engine():
    """Load the already-built index and attach the LLM.

    Called exactly once by the app (and cached there), so the model is not
    reloaded on every question.
    """
    if not os.getenv("GROQ_API_KEY"):
        raise RuntimeError(
            "GROQ_API_KEY is missing. Create a .env file (template: .env.example)."
        )

    # Important: the same embedding model as during ingest, otherwise the
    # vectors don't match and retrieval returns nonsense.
    embed_model = HuggingFaceEmbedding(model_name=config.EMBED_MODEL)
    llm = Groq(model=config.LLM_MODEL)

    # Open the existing Chroma collection (do NOT rebuild it).
    chroma_client = chromadb.PersistentClient(path=str(config.STORAGE_DIR))
    chroma_collection = chroma_client.get_or_create_collection(config.COLLECTION_NAME)
    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)

    # Reconstruct the index from the existing vector store.
    index = VectorStoreIndex.from_vector_store(
        vector_store=vector_store,
        embed_model=embed_model,
    )

    # The query engine wires up retrieval (top_k chunks) + prompt + LLM.
    return index.as_query_engine(
        llm=llm,
        similarity_top_k=config.TOP_K,
        text_qa_template=QA_PROMPT,
    )


if __name__ == "__main__":
    # Small self-test from the command line: python -m src.query
    engine = load_query_engine()
    question = "How do I make water from a stream safe to drink?"
    print(f"Question: {question}\n")
    answer = engine.query(question)
    print(f"Answer:\n{answer}\n")
    print("Source chunks used:")
    for node in answer.source_nodes:
        source = node.metadata.get("file_name", "unknown")
        print(f"  - {source} (score: {node.score:.3f})")
