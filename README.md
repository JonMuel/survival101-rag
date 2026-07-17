# Survival101 – a RAG learning project

A Retrieval-Augmented-Generation chatbot that answers survival questions purely
from a curated, copyright-clean knowledge base. Built to understand RAG systems
from the ground up.

## What is RAG (in one sentence)?

Instead of letting a language model answer freely from its memory, we first
**search** the most relevant passages from our own documents and hand them to the
model as context – so it answers with evidence and hallucinates less.

## Architecture

The project deliberately separates two phases – this is the most important design
principle and, at the same time, the prerequisite for later deployment:

```
  OFFLINE (once, when documents change)               LIVE (on every question)
  ┌─────────────────────────────────────┐             ┌──────────────────────────┐
  │ src/ingest.py                        │             │ app.py  (Streamlit chat) │
  │  1. Load    (data/raw/*)             │             │        │                 │
  │  2. Chunk                            │             │        ▼                 │
  │  3. Embed   (local, free)            │  ──────►    │ src/query.py             │
  │  4. Store   → storage/ (Chroma)      │   Index     │  Retrieval + LLM (Groq)  │
  └─────────────────────────────────────┘             └──────────────────────────┘
```

| Building block | Technology                         | Cost |
|----------------|------------------------------------|------|
| Framework      | LlamaIndex                         | free |
| Embeddings     | sentence-transformers (local, CPU) | free |
| Vector store   | Chroma (local)                     | free |
| LLM            | Groq free tier (Llama)             | free |
| UI             | Streamlit                          | free |

## Project structure

```
survival101-rag/
├── app.py               # Streamlit chat (UI)
├── requirements.txt     # Python dependencies
├── .env.example         # template for the API key (the real .env stays secret)
├── data/raw/            # source documents (PDF, Markdown, HTML ...)
├── src/
│   ├── config.py        # central settings
│   ├── ingest.py        # build the index (offline)
│   └── query.py         # RAG core (retrieval + generation)
└── storage/             # generated Chroma index (local, not in Git)
```

## Setup (Windows / VS Code)

```powershell
# 1. Create and activate a virtual environment
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Install dependencies
pip install -r requirements.txt

# 3. Provide the API key: copy .env.example to .env and paste your Groq key
#    Free key: https://console.groq.com
copy .env.example .env

# 4. Build the index (reads data/raw; first run is slow due to model download)
python -m src.ingest

# 5. Start the chat
streamlit run app.py
```

## Adding your own sources

Drop more files into `data/raw/` (PDF, Markdown, HTML, TXT) and run
`python -m src.ingest` again. Mind the legal side: safest are **public-domain**
works (e.g. the US Army Survival Manual FM 21-76), **Creative Commons** content
(check the license) or **self-written** texts.

## Deployment idea (portfolio)

Two links on your website: the **GitHub repo** (code) and a **live demo** on
[Hugging Face Spaces](https://huggingface.co/spaces) (free CPU tier). The API key
is stored there as a *secret* – never commit it to the repo. For deployment you
can push the finished `storage/` index along (adjust the rule in `.gitignore`) so
the app does not re-index on every start.

## Next learning steps

- Connect other source types (PDF, web pages) and observe chunking effects
- Evaluate retrieval quality (which chunks come back? is `TOP_K` right?)
- Add a voice interface (Groq offers Whisper speech-to-text for free)

---
*Learning project. The content is not a substitute for professional survival training.*
