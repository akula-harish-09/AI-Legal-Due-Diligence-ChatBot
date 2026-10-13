# AI Legal Due Diligence ChatBot

A local document-question-answering application for reviewing legal PDFs. Upload contracts through the Streamlit interface, retrieve relevant passages from a ChromaDB vector store, and ask Gemini to produce a structured risk summary with supporting evidence and source filenames.

> **Important:** This is an AI-assisted review tool, not legal advice. Verify all findings against the original documents and consult a qualified legal professional.

## How it works

1. The Streamlit frontend uploads PDF files to the FastAPI backend.
2. The backend extracts PDF text, splits it into overlapping chunks, embeds the chunks with `sentence-transformers/all-mpnet-base-v2`, and stores them in ChromaDB.
3. For a question, the backend retrieves up to 10 candidate chunks, reranks them with `cross-encoder/ms-marco-MiniLM-L-6-v2`, and sends the selected context to Gemini.
4. The response includes a risk level, evidence, analysis, source filenames, and a context relevance score.

## Project structure

- `frontend/app.py` — Streamlit chat UI, PDF uploader, and API client.
- `backend/main.py` — FastAPI service; provides document ingestion, retrieval, and Gemini-powered analysis.
- `ingest.py` — Optional command-line PDF ingestion script.
- `query.py` — Optional command-line question-answering script.
- `requirements.txt` — Python dependencies for both the web app and CLI scripts.
- `backend/data/` and `backend/chroma_db/` — PDFs uploaded through the web app and its vector database (created at runtime).
- `data/` and `chroma_db/` — Separate input/data locations used by the root-level CLI scripts.

**Note:** The CLI scripts and web backend use different data and ChromaDB directories. Documents ingested by one workflow are not automatically available to the other.

## Requirements

- Python 3.10 or newer is recommended.
- A Google Gemini API key with access to the model configured in `backend/main.py`.
- Internet access on first startup to download the embedding and reranking models.

## Setup

From the project root, create and activate a virtual environment, then install the dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Set `GOOGLE_API_KEY` in the process environment or create a local `.env` file in the project root containing:

```text
GOOGLE_API_KEY=your_google_api_key_here
```

Do not commit `.env` or share its contents. If a real API key has ever been exposed, revoke it and create a replacement.

## Run the web app

Start the backend in one PowerShell terminal from the project root:

```powershell
Set-Location backend
uvicorn main:app --reload
```

Start the frontend in a second terminal from the project root:

```powershell
streamlit run frontend/app.py
```

Open the local URL printed by Streamlit (usually `http://localhost:8501`). The frontend expects the backend at `http://localhost:8000`. FastAPI's interactive API documentation is available at `http://localhost:8000/docs`.

Upload one or more PDFs in the sidebar, then enter a question or choose one of the sample questions. The backend also exposes:

- `GET /health` — backend health check.
- `GET /documents` — list ingested PDF filenames.
- `POST /ingest` — ingest a PDF upload.
- `POST /query` — return a complete answer.
- `POST /query/stream` — stream an answer using server-sent events.

## Optional command-line workflow

Run these from the project root:

```powershell
python ingest.py
python query.py
```

Place source PDFs in `data/` before running `ingest.py`. These commands use the root-level `data/` and `chroma_db/` directories, not the backend's directories. The CLI query script also requires `GOOGLE_API_KEY` to be set in the environment; its model configuration is in `query.py`.

## Configuration and local data

The backend reads `CHROMA_DIR` and `DATA_DIR` environment variables, defaulting to `./chroma_db` and `./data` relative to the backend's working directory. Uploaded PDFs, vector databases, downloaded model caches, virtual environments, and secrets are local/generated files and should not be committed.

The backend currently enables permissive CORS for development. Review and restrict this before exposing the service beyond a trusted local environment.
