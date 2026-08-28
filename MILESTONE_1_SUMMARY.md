# Milestone 1: Baseline RAG - Complete ✓

## Overview

Milestone 1 establishes the foundation of RAGScope with a complete baseline RAG pipeline: PDF ingestion, chunking, embedding, vector storage, retrieval, and LLM-based answer generation with citations.

## What Was Built

### 1. Document Ingestion Pipeline

**PDF Parser (`app/ingestion/parser.py`)**
- Extracts text from PDF files using PyPDF
- Preserves page numbers for each page
- Returns structured page objects with text and metadata

**Text Chunker (`app/ingestion/chunker.py`)**
- Splits pages into configurable chunks (default: 1000 chars, 200 overlap)
- Uses LangChain's RecursiveCharacterTextSplitter
- Preserves metadata (document name, page number)
- Generates unique chunk IDs

**Indexer (`app/ingestion/indexer.py`)**
- Generates embeddings using OpenAI text-embedding-3-small
- Stores vectors in Pinecone with metadata
- Batch processing for efficiency
- Saves chunks to JSON for BM25 indexing

### 2. Retrieval System

**Vector Retriever (`app/retrieval/vector.py`)**
- Embeds queries using same model as indexing
- Performs cosine similarity search in Pinecone
- Returns top-K chunks with scores and metadata
- Structured result objects for downstream processing

### 3. Generation System

**Prompts (`app/generation/prompts.py`)**
- System prompt instructs LLM to use only provided context
- Citation format with chunk references
- Fallback message for insufficient information

**Generator (`app/generation/generator.py`)**
- Uses OpenAI GPT-4o-mini for generation
- Formats context with chunk metadata
- Extracts citations from generated answer
- Returns answer with source attribution

### 4. API Layer

**Documents API (`app/api/documents.py`)**
- `POST /documents/upload` - Upload PDF files
- `POST /documents/index/{id}` - Index uploaded documents
- `GET /documents` - List all documents
- `DELETE /documents/{id}` - Delete documents

**Query API (`app/api/query.py`)**
- `POST /query` - Execute RAG query
- Returns answer, sources, and latency

**Traces API (`app/api/traces.py`)**
- `GET /traces` - List recent traces
- `GET /traces/{id}` - Get trace details

### 5. Database Schema

**Models (`app/models/schemas.py`)**
- `Document` - Tracks uploaded documents
- `Trace` - Stores query execution traces
- Pydantic models for request/response validation

### 6. Configuration

**Settings (`app/config/settings.py`)**
- Centralized configuration using Pydantic Settings
- Environment variable support
- Defaults for all parameters

## API Endpoints

### Documents
```bash
POST /documents/upload          # Upload PDF
POST /documents/index/{id}      # Index document
GET /documents                  # List documents
DELETE /documents/{id}          # Delete document
```

### Query
```bash
POST /query
{
  "question": "What is the refund policy?",
  "top_k": 5
}
```

Response:
```json
{
  "answer": "The refund policy allows returns within 30 days.",
  "sources": [
    {
      "document": "policy.pdf",
      "page": 4,
      "chunk_id": "policy_p4_c1",
      "similarity_score": 0.91
    }
  ],
  "latency_ms": 820,
  "trace_id": "abc123..."
}
```

## Files Created

### Backend
- `app/ingestion/parser.py` - PDF parsing
- `app/ingestion/chunker.py` - Text chunking
- `app/ingestion/indexer.py` - Embedding and indexing
- `app/retrieval/vector.py` - Vector retrieval
- `app/generation/prompts.py` - LLM prompts
- `app/generation/generator.py` - Answer generation
- `app/api/documents.py` - Document endpoints
- `app/api/query.py` - Query endpoint
- `app/api/traces.py` - Trace endpoints
- `app/models/schemas.py` - Data models
- `app/config/settings.py` - Configuration
- `app/database/session.py` - Database setup
- `app/main.py` - FastAPI application

### Tests
- `tests/test_parser.py` - Parser tests
- `tests/test_chunker.py` - Chunker tests
- `tests/test_vector_retrieval.py` - Retrieval tests
- `tests/test_api.py` - API tests

### Configuration
- `pyproject.toml` - Dependencies
- `.env.example` - Environment template
- `docker-compose.yml` - PostgreSQL setup

## Dependencies

```toml
fastapi>=0.115.0           # Web framework
uvicorn[standard]>=0.32.0  # ASGI server
pydantic>=2.9.0            # Data validation
pypdf>=5.0.0               # PDF parsing
langchain>=0.3.0           # LLM orchestration
langchain-openai>=0.3.0    # OpenAI integration
pinecone>=5.0.0            # Vector database
sqlalchemy>=2.0.0          # ORM
asyncpg>=0.29.0            # PostgreSQL driver
```

## Usage

### Start Services
```bash
# Start PostgreSQL
docker-compose up -d postgres

# Start backend
cd backend
uv run uvicorn app.main:app --reload
```

### Upload and Query
```bash
# Upload document
curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"

# Index document
curl -X POST http://localhost:8000/documents/index/{id}

# Query
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What is the refund policy?"}'
```

## Architecture

```
PDF → Parser → Chunker → Embeddings → Pinecone
                                      ↓
Query → Embed → Vector Search → Top-K Chunks → LLM → Answer + Citations
```

## Verification

```bash
cd backend
python3 verify_milestone1.py
```

---

**Status**: ✓ Complete  
**Date**: August 5, 2026  
**Components**: 13 modules, 4 test files, 4 API endpoints
