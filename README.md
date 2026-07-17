# 🏕️ Survival101 – ein RAG-Lernprojekt

Ein Retrieval-Augmented-Generation-Chatbot, der Fragen rund ums Thema Survival
ausschließlich auf Basis einer kuratierten, quellenrechtlich sauberen
Wissensbasis beantwortet. Gebaut, um RAG-Systeme von Grund auf zu verstehen.

## Was ist RAG (in einem Satz)?

Statt ein Sprachmodell frei aus seinem Gedächtnis antworten zu lassen, **suchen**
wir zuerst die relevantesten Textstellen aus eigenen Dokumenten heraus und geben
sie dem Modell als Kontext mit – so antwortet es belegbar und halluziniert weniger.

## Architektur

Das Projekt trennt bewusst zwei Phasen – das ist das wichtigste Designprinzip
und zugleich die Voraussetzung fürs spätere Deployment:

```
  OFFLINE (einmalig, wenn sich Dokumente ändern)      LAUFEND (bei jeder Frage)
  ┌─────────────────────────────────────┐             ┌──────────────────────────┐
  │ src/ingest.py                        │             │ app.py  (Streamlit-Chat) │
  │  1. Laden   (data/raw/*)             │             │        │                 │
  │  2. Chunking                         │             │        ▼                 │
  │  3. Embedding (lokal, gratis)        │  ──────►    │ src/query.py             │
  │  4. Speichern → storage/ (Chroma)    │   Index     │  Retrieval + LLM (Groq)  │
  └─────────────────────────────────────┘             └──────────────────────────┘
```

| Baustein     | Technologie                          | Kosten |
|--------------|--------------------------------------|--------|
| Framework    | LlamaIndex                           | gratis |
| Embeddings   | sentence-transformers (lokal, CPU)   | gratis |
| Vektor-Store | Chroma (lokal)                       | gratis |
| LLM          | Groq Free-Tier (Llama)               | gratis |
| Oberfläche   | Streamlit                            | gratis |

## Projektstruktur

```
survival101-rag/
├── app.py               # Streamlit-Chat (Oberfläche)
├── requirements.txt     # Python-Abhängigkeiten
├── .env.example         # Vorlage für den API-Key (echte .env bleibt geheim)
├── data/raw/            # Quelldokumente (PDF, Markdown, HTML ...)
├── src/
│   ├── config.py        # zentrale Stellschrauben
│   ├── ingest.py        # Index bauen (offline)
│   └── query.py         # RAG-Kern (Retrieval + Generation)
└── storage/             # erzeugter Chroma-Index (lokal, nicht im Git)
```

## Setup (Windows / VS Code)

```powershell
# 1. Virtuelle Umgebung anlegen und aktivieren
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# 2. Abhängigkeiten installieren
pip install -r requirements.txt

# 3. API-Key hinterlegen: .env.example zu .env kopieren und Groq-Key eintragen
#    Kostenloser Key: https://console.groq.com
copy .env.example .env

# 4. Index bauen (liest data/raw, dauert beim ersten Mal wegen Modell-Download)
python -m src.ingest

# 5. Chat starten
streamlit run app.py
```

## Eigene Quellen hinzufügen

Lege weitere Dateien in `data/raw/` ab (PDF, Markdown, HTML, TXT) und führe
`python -m src.ingest` erneut aus. Achte auf die Rechtslage: Am sichersten sind
**gemeinfreie** Werke (z. B. das US Army Survival Manual FM 21-76), **Creative-
Commons**-Inhalte (Lizenz beachten) oder **selbst geschriebene** Texte.

## Deployment-Idee (Portfolio)

Zwei Links auf der Webseite: das **GitHub-Repo** (Code) und eine **Live-Demo**
auf [Hugging Face Spaces](https://huggingface.co/spaces) (kostenloses CPU-Tier).
Der API-Key wird dort als *Secret* hinterlegt – niemals ins Repo committen. Für
das Deployment kann man den fertigen `storage/`-Index mitpushen (dazu die Regel
in `.gitignore` anpassen), damit die App nicht bei jedem Start neu indexieren muss.

## Nächste Lernschritte

- Andere Quellentypen anbinden (PDF, Webseiten) und Chunking-Effekte beobachten
- Retrieval-Qualität bewerten (welche Chunks kommen zurück? Stimmt `TOP_K`?)
- Voice-Interface ergänzen (Groq bietet Whisper-Speech-to-Text gratis)

---
*Lernprojekt. Die Inhalte ersetzen keine professionelle Survival-Ausbildung.*
