# Milestone 3: BM25 + Hybrid Retrieval - Complete ✓

## Overview

Milestone 3 adds lexical retrieval capabilities to complement the existing vector-based semantic search. This enables RAGScope to handle keyword-specific queries (error codes, product names, technical terms) that vector search might miss.

## What Was Built

### 1. BM25 Retriever (`app/retrieval/bm25.py`)
- Implements BM25Okapi algorithm for lexical keyword matching
- Case-insensitive matching (lowercases both corpus and queries)
- Builds index from document chunks stored during indexing
- Returns scored results with BM25 relevance scores

**Key Features:**
- Lazy loading: BM25 index is built on first use
- Persistent storage: Chunks saved to `bm25_chunks.json` during indexing
- Efficient retrieval using numpy vectorization

### 2. Hybrid Retriever (`app/retrieval/hybrid.py`)
- Combines vector and BM25 retrieval results
- Implements Reciprocal Rank Fusion (RRF) for score combination
- Configurable candidate_k (retrieves more candidates before fusion)
- Returns top_k results after fusion

**RRF Formula:**
```
score(d) = Σ 1/(k + rank_i(d))
```
Where k=60 (standard constant) and rank_i(d) is the rank of document d in retrieval method i.

### 3. Retrieval Factory (`app/retrieval/factory.py`)
- Unified interface for all retrieval modes
- Supports three modes:
  - `vector`: Pure semantic search
  - `bm25`: Pure lexical search
  - `hybrid`: Combined search with RRF
- Lazy initialization of BM25 index
- Type-safe retrieval with proper result types

### 4. Updated Components

**Schemas (`app/models/schemas.py`):**
- Added `retrieval_mode` field to `QueryRequest`
- Added `retrieval_mode` to `Trace` model
- Updated `TraceResponse` to include retrieval mode

**Query Endpoint (`app/api/query.py`):**
- Uses `RetrievalFactory` instead of direct `VectorRetriever`
- Handles different chunk types (RetrievedChunk, BM25Result, HybridChunk)
- Passes retrieval mode to tracer
- Converts all chunk types to common format for generation

**Indexer (`app/ingestion/indexer.py`):**
- Saves chunks to `bm25_chunks.json` during indexing
- Prevents duplicate chunks using chunk_id tracking
- Enables BM25 index building without re-parsing PDFs

**Tracer (`app/tracing/tracer.py`):**
- Accepts `retrieval_mode` parameter
- Handles different chunk types with isinstance checks
- Captures vector_score, bm25_score, and rrf_score appropriately
- Stores retrieval mode in trace for later analysis

## Technical Decisions

### Why BM25?
- **Keyword matching**: Vector search can miss exact matches for error codes, product names, IDs
- **Complementary**: BM25 excels where embeddings struggle (exact terminology)
- **Fast**: BM25 is computationally efficient for large corpora
- **Interpretable**: BM25 scores are easier to understand than embedding similarity

### Why Reciprocal Rank Fusion?
- **Simple**: Easy to understand and implement
- **Effective**: Works well in practice for combining rankings
- **Score-agnostic**: Doesn't require score normalization between methods
- **Robust**: Handles cases where one method returns no results

### Why Store Chunks in JSON?
- **Persistent**: BM25 index can be rebuilt without re-parsing PDFs
- **Fast**: Loading JSON is faster than querying Pinecone for all documents
- **Simple**: No additional database or service needed
- **Portable**: Easy to backup and migrate

## API Changes

### Query Endpoint
```json
POST /query
{
  "question": "What is error code ERR-404?",
  "top_k": 5,
  "retrieval_mode": "hybrid"  // NEW: "vector" | "bm25" | "hybrid"
}
```

### Trace Response
Now includes `retrieval_mode` and separate scores:
```json
{
  "retrieval_mode": "hybrid",
  "retrieval_results": {
    "chunks": [
      {
        "chunk_id": "...",
        "vector_score": 0.85,
        "bm25_score": 12.5,
        "rrf_score": 0.0328,
        "final_rank": 1
      }
    ]
  }
}
```

## Testing

### Test Coverage
- **BM25 Tests** (`tests/test_bm25.py`):
  - Index building
  - Retrieval with scoring
  - Empty index handling
  - No-match scenarios
  
- **Hybrid Tests** (`tests/test_hybrid.py`):
  - RRF fusion logic
  - RRF formula verification
  - Top-k limiting
  - Candidate-k configuration

### Test Results
```
tests/test_bm25.py::test_bm25_retriever_build_index PASSED
tests/test_bm25.py::test_bm25_retriever_retrieve PASSED
tests/test_bm25.py::test_bm25_retriever_empty_index PASSED
tests/test_bm25.py::test_bm25_retriever_no_matches PASSED
tests/test_hybrid.py::test_hybrid_retriever_rrf PASSED
tests/test_hybrid.py::test_hybrid_retriever_rrf_formula PASSED
tests/test_hybrid.py::test_hybrid_retriever_top_k PASSED
tests/test_hybrid.py::test_hybrid_retriever_candidate_k PASSED
```

## Dependencies Added

```toml
"rank-bm25>=0.2.2"  # BM25 implementation
"numpy<2.0.0"       # Required by rank-bm25 (v2.x incompatible)
```

## Files Created/Modified

### New Files
- `app/retrieval/bm25.py` - BM25 retriever implementation
- `app/retrieval/hybrid.py` - Hybrid retriever with RRF
- `app/retrieval/factory.py` - Retrieval mode factory
- `tests/test_bm25.py` - BM25 unit tests
- `tests/test_hybrid.py` - Hybrid retrieval tests
- `verify_milestone3.py` - Verification script

### Modified Files
- `app/models/schemas.py` - Added retrieval_mode fields
- `app/api/query.py` - Integrated retrieval factory
- `app/ingestion/indexer.py` - Added BM25 chunk storage
- `app/tracing/tracer.py` - Handle multiple chunk types
- `pyproject.toml` - Added rank-bm25 and numpy dependencies
- `README.md` - Updated with retrieval modes and trace endpoints

## Usage Examples

### Vector Search (Default)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "How do I request a refund?",
    "top_k": 5,
    "retrieval_mode": "vector"
  }'
```

### BM25 Search (Keyword Matching)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is error code ERR-404?",
    "top_k": 5,
    "retrieval_mode": "bm25"
  }'
```

### Hybrid Search (Best of Both)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What are the API rate limits?",
    "top_k": 5,
    "retrieval_mode": "hybrid"
  }'
```

## Performance Considerations

### BM25 Index Building
- **First query**: Loads and indexes all chunks (~1-2 seconds for 1000 chunks)
- **Subsequent queries**: Uses cached index (instant)
- **Memory**: Stores all chunk texts in memory

### Hybrid Retrieval
- **Retrieves**: candidate_k=20 from each method (configurable)
- **Fuses**: Combines results using RRF
- **Returns**: top_k results after fusion
- **Latency**: ~2x vector retrieval (retrieves from both methods)

## Next Steps (Milestone 4)

With hybrid retrieval in place, the next milestone will add:
- Cross-encoder reranking for improved accuracy
- Reranker scores captured in traces
- Configurable reranking in retrieval pipeline
- Comparison of retrieval vs reranked results

## Verification

Run the verification script:
```bash
cd backend
uv run python verify_milestone3.py
```

All tests:
```bash
uv run pytest tests/test_bm25.py tests/test_hybrid.py tests/test_tracing.py -v
```

---

**Status**: ✓ Complete  
**Date**: August 5, 2026  
**Test Coverage**: 8 new tests, all passing
