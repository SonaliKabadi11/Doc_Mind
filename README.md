# DocMind

A production-grade RAG (Retrieval - Augmented Generation) system that lets useres upload a document and ask natural-languaue wuestions about it, with answers grounded in the document's actual content and cited back to source passage.

## Problem Statement
Users often need to extract specific information from long documents (research papers, contracts, manuals, internal wikis) without reading them in full. DocMind lets a user upload a document and ask natural-language questions about it, returning answers grounded in the document's actual content — with citations back to the source — rather than relying on a model's general (and potentially incorrect or outdated) knowledge.


## Arcitecture

```
Ingestion (offline, per document)

PDF/TXT -> text extraction -> embed (pretrained, frozen) -> store in vector DB (scoped to document_id)

Query (online, per request)

Query -> embed -> retrieve top-k chunkns (scoped to document_id) -> LLM generation -> grounded answer + citations

```


### File Structure
```text
docmind-rag/
├── backend/
│   ├── app/
│   │   ├── core/          # settings, logging
│   │   ├── api/           # FastAPI routes
│   │   ├── ingestion/     # PDF parsing, chunking, embedding
│   │   ├── retrieval/     # vector search, reranking
│   │   ├── generation/    # LLM provider interface + implementations
│   │   └── evaluation/    # precision@k / recall@k harness
│   ├── tests/
│   ├── Dockerfile
│   └── requirements.txt
├── frontend/               # Vue/React chat UI (Phase 4)
├── docker-compose.yml
├── .github/workflows/      # CI
└── README.md
```

## Design Decisions

### In-memory Database
Chose an in-memory vector store for v1 to avoid managing external DB infrastructure on a limited-time portfolio project; documents don't persist across restarts. A v2 iteration would swap in pgvector or a managed vector DB for persistence — the retrieval interface is designed so this swap doesn't touch the rest of the pipeline.