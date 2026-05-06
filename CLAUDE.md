# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Indonesian legal Q&A application using a dual-pipeline RAG architecture with amendment-awareness. Runs KG-RAG (knowledge graph via Neo4j) and Naive-RAG (vector search) in parallel, with optional amendment processing, then aggregates the results.

## Running the App

```bash
pip install -r requirements.txt
python app.py           # Web server on http://0.0.0.0:5000
python multi-agent.py   # CLI mode (no web UI)
```

## Architecture

### Backend (`app.py` + `multi-agent.py`)
- **app.py**: Flask server with SSE streaming. `POST /api/query` submits questions; `GET /api/events/<session_id>` streams pipeline progress in real-time.
- **multi-agent.py**: Core RAG engine (~2400 lines). All LLM calls go through `gemini_call()` with rate limiting. Neo4j stores pre-embedded chunks, triples, entities, and amendment chains.

### Frontend (`static/` + `templates/`)
- Single-page app with 5 tabs: KG-RAG, Naive-RAG, Amendment Processing, Aggregator, Final Answer.
- Real-time updates via SSE; iteration cards show query → answer → judge verdict → modified query flow.

### Key Data Flow
```
User Question
    │
    ├── KG-RAG (parallel thread)  ──► Agent 1/1b extraction → triple/entity retrieval → Answer Judge loop
    │
    ├── Naive-RAG (parallel thread) ──► vector search → Answer Judge loop
    │
    ├── Amendment Processing (before or after aggregation)
    │
    └── Aggregator Agent → Final Answer
```

## Neo4j Data Model
- `TextChunk` nodes with pre-stored embeddings (index: `chunk_embedding_index`) -- do NOT re-embed at runtime
- `Triple` nodes (index: `triple_vec`) and `Entity` nodes (indexes: `document_vec`, `content_vec`, `expression_vec`)
- `AMD_UndangUndang` nodes with amendment relationships (`AMD_DIUBAH_DENGAN`, `AMD_DICABUT_DENGAN`)
- ChunkStore fallback uses pickle files if Neo4j lookup fails

## Agent Roles
| Agent | Purpose |
|---|---|
| Agent 1 / 1b | Extract entities, predicates, and triples from query |
| Agent 2 (Answerer) | Generate answers from context |
| Agent AJ (Answer Judge) | Evaluate answer quality, provide feedback |
| Agent QM (Query Modifier) | Rewrite query based on judge feedback |
| Aggregator Agent | Choose KG-RAG, Naive-RAG, or merge |
| Amendment Relevance Judge | Judge if amending laws affect the answer |
| Amendment Integration Agent | Integrate new provisions into answers |

## Configuration (.env)
| Variable | Default | Notes |
|---|---|---|
| `GOOGLE_API_KEY` | required | Gemini API key |
| `NEO4J_URI` | `bolt://localhost:7687` | Neo4j connection |
| `NEO4J_USER` / `NEO4J_PASS` | `neo4j` / `password` | |
| `GEN_MODEL` | `models/gemini-2.5-flash-lite` | Gemini generation model |
| `EMBED_MODEL` | -- | Deprecated; Gemini embedding no longer used |
| `EMBED_MODEL_NAME` | `BAAI/bge-m3` | Local embedding model |
| `EMBED_DIMENSION` | `1024` | BAAI/bge-m3 embedding dimension (constant) |
| `MAX_ANSWER_JUDGE_ITERS` | `3` | Max iterations per pipeline |
| `TOP_K_CHUNKS` | `40` | Chunks retrieved per pipeline |
| `LLM_EMBED_MAX_CONCURRENCY` | `8` | Embedding concurrency limit |
| `LLM_GEN_QPS` | `1.0` | Generation QPS rate limit |

## Important Notes
- **Pre-stored embeddings**: Only query text, entities, and triples are embedded at runtime using a local BAAI/bge-m3 model. Chunk embeddings are pre-stored in Neo4j (also BAAI/bge-m3, 1024-dim).
- **Caching**: `_EMB_CACHE` caches embedding results (SHA256 key); amendment chain traversals are cached in memory.
- **Parallelism**: Both pipelines run in background threads; `app.py` monkey-patches `log()` to broadcast to SSE clients.
- **Amendment modes**: `before_aggregation` (default) or `after_aggregation` controlled via `AMENDMENT_MODE` param in the API.
