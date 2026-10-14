# DocMind

DocMind is a document question-answering API built around retrieval-augmented generation (RAG). Upload a PDF, retrieve relevant passages, or ask a question and receive an answer with citations to source pages.

## Features

- PDF text extraction and page-aware chunking
- Semantic embeddings using `sentence-transformers/all-MiniLM-L6-v2`
- Per-document in-memory vector search
- Gemini access through its OpenAI-compatible API
- Similarity threshold to refuse answers when retrieved passages are weak matches
- Citation validation so answers only cite retrieved passages
- FastAPI API with interactive documentation

## How It Works

```text
PDF upload -> extract text by page -> split into chunks -> embed -> in-memory store

Question -> embed -> retrieve top matches -> apply relevance threshold
					-> generate grounded answer -> validate citations
```

The vector store is scoped by document ID. Stored documents are lost when the server restarts; uploaded PDFs are not persisted.

## Requirements

- Python 3.12 or newer
- A Google AI API key for Gemini
- Internet access the first time the embedding model is downloaded and when calling Gemini

## Run Locally

From the repository root, create and activate a virtual environment, then install the backend dependencies:

```powershell
cd backend
py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

Create `backend/.env`:

```dotenv
GOOGLE_API_KEY=your-google-ai-api-key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/openai/
```

Do not commit this file. `.env` files are ignored by Git. The settings load `.env` relative to the process working directory, so start the application from `backend`:

```powershell
python -m uvicorn app.main:app --reload
```

The API is available at <http://127.0.0.1:8000>. Interactive Swagger documentation is at <http://127.0.0.1:8000/docs>.

## API

| Method | Endpoint | Description |
| --- | --- | --- |
| `GET` | `/health` | Check that the API is responding |
| `POST` | `/documents` | Upload a PDF as multipart form field `file` |
| `POST` | `/documents/{document_id}/query` | Return the most relevant passages |
| `POST` | `/documents/{document_id}/ask` | Generate an answer with validated citations |
| `DELETE` | `/documents/{document_id}` | Remove a document from the in-memory store |

Query and ask requests accept JSON with a required `question` and optional `top_k` (default `5`, range `1`-`20`). For example:

```json
{
	"question": "What does the document say about model evaluation?",
	"top_k": 5
}
```

Upload a PDF with PowerShell using `curl.exe`:

```powershell
curl.exe -F "file=@report.pdf" http://127.0.0.1:8000/documents
```

Use the returned `document_id` with the query, ask, and delete endpoints. The ask response includes the answer, whether it is grounded, and citation details such as page number and source text.

Common error responses include `404` for an unknown document, `413` for a file over the configured size limit, `415` for a non-PDF upload, `422` for ingestion or request validation errors, and `502`/`504` for LLM errors and timeouts.

## Configuration

Settings can be supplied through environment variables or `backend/.env` when launched from the backend directory.

| Variable | Purpose | Default |
| --- | --- | --- |
| `GOOGLE_API_KEY` | Gemini API credential | Required for application startup |
| `GEMINI_BASE_URL` | OpenAI-compatible Gemini endpoint | `https://generativelanguage.googleapis.com/v1beta/openai/` |
| `LLM_MODEL` | Chat model name | `gemini-3.1-flash-lite` |
| `LLM_TEMPERATURE` | Generation temperature | `0.0` |
| `LLM_MAX_TOKENS` | Maximum generated tokens | `1024` |
| `LLM_TIMEOUT_SECONDS` | LLM request timeout | `30.0` |
| `LLM_MAX_RETRIES` | LLM request retries | `2` |
| `EMBEDDING_MODEL` | Sentence Transformers model | `sentence-transformers/all-MiniLM-L6-v2` |
| `CHUNK_SIZE` | Text chunk size | `1000` |
| `CHUNK_OVERLAP` | Overlap between chunks | `200` |
| `MAX_UPLOAD_MB` | Maximum PDF upload size | `20` |
| `SIMILARITY_THRESHOLD` | Minimum retrieval score to generate an answer | `0.10` |
| `MAX_CONTEXT_CHARS` | Maximum context sent to the LLM | `12000` |
| `APP_NAME`, `ENVIRONMENT`, `LOG_LEVEL` | Application metadata and logging | `DocMind`, `development`, `INFO` |

## Tests and Lint

From `backend`:

```powershell
python -m pytest -q
ruff check .
```

The tests use deterministic fake embeddings and an LLM stub for API and RAG behavior; they do not make live Gemini requests.

## Project Layout

```text
backend/
	app/
		api/          FastAPI routes and request/response schemas
		config/       Application settings
		generation/   LLM provider, prompts, and citation validation
		ingestion/    PDF extraction, chunking, and embeddings
		retrieval/    Retrieval interface and in-memory vector store
		services/     RAG orchestration
	tests/          Unit and API tests
frontend/         Frontend placeholder; no UI implemented yet
docker-compose.yml
```

## Current Limitations

- Documents and embeddings exist only in process memory and disappear on restart.
- There is no authentication or authorization; do not expose this API publicly as-is.
- Docker Compose is a scaffold, not a supported run path yet: its backend Dockerfile is empty and the compose file references a database service that is not defined.
- The frontend directory is currently a placeholder.