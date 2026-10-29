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
python -m pip install -r requirements-dev.txt
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

## Retrieval Evaluation

The retrieval evaluation compares chunking configurations using evidence-based Hit@k and MRR metrics. See the [October 23, 2026 evaluation report](backend/eval/retrieval-evaluation-2026-10-23.md) for the results, methodology, and proposed reranking experiment.

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


## Evaluation

DocMind is evaluated against a labeled question set instead of being judged by demos. Every change to chunking, retrieval or prompting is recorded with before/after numbers in [`docs/evaluation/`](docs/evaluation/README.md).

**Where things stand** (dev split, retrieval stage only): with the default settings (1,000-character chunks, 200 overlap, top 5), a chunk containing the answer's evidence is retrieved for **48%** of answerable questions. Retrieving the top 10 instead raises that to **70%** (+10 of 44 questions). About 1 in 5 answerable questions is still missed even within the top 20, so **retrieval is the current bottleneck**. Generation metrics (false refusals, faithfulness) are the next milestone.

### Method

- **Documents (4):** *Alice's Adventures in Wonderland* (public domain), the NIST AI Risk Management Framework 1.0 (NIST AI 100-1), the NIST AI RMF Playbook, and Divoli, Nakov & Hearst (2012), *Do peers see more in a paper than its authors?* The PDFs are not redistributed; sources and licences are listed in [`docs/evaluation/README.md`](docs/evaluation/README.md).
- **Questions:** 100 hand-written questions (25 per document). Answerable questions carry verbatim evidence quotes from the document; unanswerable ones are off-topic, hard negatives or false-premise questions. Types: fact, entity, numeric, table, multi-chunk.
- **Relevance is defined by evidence text, not chunk IDs.** A retrieved chunk counts as relevant if it contains a labeled quote (matching ignores case, whitespace, punctuation and PDF extraction noise), so labels stay valid when chunk size, overlap or the embedding model change.
- **Splits.** A stable hash of the question ID assigns about 60% of questions to *dev* (used for every tuning decision) and about 40% to *test* (one final run, never used for tuning).
- **Metrics.**
  - *Hit rate@k*: a relevant chunk is in the top k.
  - *MRR*: reciprocal rank of the first relevant chunk (computed over the top 20).
  - *Intact*: share of questions whose evidence is not cut by chunking, i.e. the ceiling for hit rate.
  - Classic precision and recall are not reported. With evidence-level labels there is no stable count of "all relevant chunks" (it changes with chunk size and overlap), and with one or two relevant chunks per question, precision is capped near 1/k.

### Baseline: chunking grid

Dev split, 44 answerable questions, `all-MiniLM-L6-v2` embeddings, 3 chunk sizes x 3 overlaps. Counts out of 44. Full report: [R0](docs/evaluation/reports/2026-10-28-baseline-chunking-grid.md).

| Size / overlap | Chunks | Intact | Hit@5 | Hit@10 | Hit@20 | MRR |
|---|---:|---:|---:|---:|---:|---:|
| 500 / 0 | 1617 | 42 | 17 (39%) | 24 (55%) | 27 (61%) | 0.317 |
| 500 / 100 | 1809 | 41 | 16 (36%) | 24 (55%) | 30 (68%) | 0.319 |
| 500 / 200 | 2305 | 43 | 15 (34%) | 22 (50%) | 28 (64%) | 0.287 |
| 1000 / 0 | 852 | 43 | 15 (34%) | 27 (61%) | 33 (75%) | 0.261 |
| 1000 / 100 | 867 | 43 | 17 (39%) | 28 (64%) | 32 (73%) | 0.302 |
| **1000 / 200** | 933 | 43 | 21 (48%) | 31 (70%) | 36 (82%) | 0.344 |
| 1500 / 0 | 592 | 41 | 24 (55%) | 27 (61%) | 33 (75%) | 0.364 |
| 1500 / 100 | 593 | 43 | 27 (61%) | 29 (66%) | 34 (77%) | 0.390 |
| 1500 / 200 | 611 | 43 | 25 (57%) | 29 (66%) | 36 (82%) | 0.389 |

### Findings

1. **The level is low and does not saturate.** Hit@5 is 34-61%, Hit@10 is 50-70%, Hit@20 is 61-82%, and Hit@1 is only 16-27%. The right chunk is often retrieved but not ranked first.
2. **Chunk size and overlap matter less than they appear.** Larger chunks look better at equal k mostly because they retrieve more text (k=5 is about 2,500 characters at size 500 but 7,500 at size 1500). At matched context size the configurations land within a few questions of each other, which is inside the noise for 44 questions. Overlap has no consistent effect.
3. **Chunking is not hiding the evidence.** `intact` is 41-43 of 44 in every configuration. One question (`nist-ai-rmf-019`) has an evidence quote that is not present in the extracted text and is a known dataset issue.
4. **Where the misses are (1000/200).** Of the 23 questions with no relevant chunk in the top 5, 10 sit at ranks 6-10 and 5 at ranks 11-20 (a reranker or a larger `top_k` can recover these), while 8 never reach the top 20 (candidate generation fails; one is the bad quote).
5. **Difficulty varies by document.** Alice is the easiest (Hit@10 12/12 at 1000/200). The research paper and the NIST AI RMF are the hardest (6/12 and 5/9). Numeric and exact-fact questions look weakest, but there are only 2 numeric questions, so this is a lead to test, not a result.

### Improvement log

Every row is measured on the same dataset version and split. Full history, rules and templates: [`docs/evaluation/`](docs/evaluation/README.md).

| ID | Change | App `top_k` | Hit@`top_k` (n=44) | Hit@20 | MRR | Status |
|---|---|---:|---:|---:|---:|---|
| R0 | Baseline: 1000/200 chunks, top 5 | 5 | 21 (48%) | 36 (82%) | 0.344 | Baseline |
| R1 | `top_k` 5 to 10 | 10 | 31 (70%) | 36 (82%) | 0.344 | Proposed: roughly doubles retrieved context; generation check pending |
| R2 | Hybrid search (BM25 + dense) | - | - | - | - | Planned |
| R3 | Cross-encoder reranker over top 20 | - | - | - | - | Planned |
| R4 | Embedding model comparison | - | - | - | - | Planned |
| R5 | Prompt variants (generation: false refusals, faithfulness) | - | - | - | - | Planned |
| Final | Single run on the test split | - | - | - | - | Planned |

### Limitations

- **Small dataset.** 44 answerable dev questions: one question is 2.3 points, and a hit rate near 50% carries a sampling error of roughly 7 points. Differences of a few questions are noise.
- **Labels are lower bounds.** A chunk that answers the question in different words than the labeled quote counts as a miss.
- **Dev numbers only.** The test split has not been used. Headline numbers will come from one final test run.
- **Retrieval only so far.** Retrieval gains are not application gains until generation metrics (false-refusal rate, hallucination rate, faithfulness) confirm them.
- **One embedding model, no hybrid search or reranking yet.**
- **Equal-k comparisons across chunk sizes are not equal-context comparisons.** See finding 2.

### Reproduce

From `backend/`:

```powershell
# 1. Check that every evidence quote exists in its PDF
python -m app.evaluation.validate_dataset --docs evals\docs --questions evals\questions

# 2. Run the chunking grid on the dev split (--detail adds per-document, per-type and per-question misses)
python -m app.evaluation.run_retrieval --docs evals\docs --questions evals\questions --detail
```

Place the PDFs at `backend/evals/docs/<doc>.pdf` (`alice`, `nist_ai_rmf`, `nist_playbook`, `peers_paper`) and the question files at `backend/evals/questions/*.jsonl`.
