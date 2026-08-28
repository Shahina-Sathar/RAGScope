# Milestone 4: Reranking - Complete ✓

## Overview

Milestone 4 adds cross-encoder reranking to improve retrieval accuracy. After initial retrieval (vector, BM25, or hybrid), a cross-encoder model re-scores the candidate chunks with a more sophisticated query-document interaction model, producing a final ranked list for generation.

## What Was Built

### 1. Cross-Encoder Reranker (`app/retrieval/reranker.py`)

**Key Components:**
- `RerankedChunk` dataclass - Stores reranked results with both original and reranker scores
- `CrossEncoderReranker` class - Wraps sentence-transformers CrossEncoder model
- Lazy model loading - Model only loads when first needed (faster startup)
- Default model: `cross-encoder/ms-marco-MiniLM-L-6-v2` (fast, accurate)

**How It Works:**
1. Takes query + list of candidate chunks from any retrieval method
2. Creates query-document pairs
3. Cross-encoder predicts relevance score for each pair
4. Sorts chunks by reranker score (descending)
5. Returns top_k chunks with full metadata

**Why Cross-Encoder?**
- **Bi-encoder** (embeddings): Encodes query and document separately, then compares vectors. Fast but less accurate.
- **Cross-encoder**: Encodes query-document pair together, allowing full attention between them. Slower but much more accurate.

### 2. Updated Query Pipeline (`app/api/query.py`)

**New Flow:**
```
Query → Retrieve (candidate_k) → [Rerank] → Top (reranker_top_k) → Generate
```

**Logic:**
- If `reranker_enabled=false`: Retrieve `top_k` chunks directly
- If `reranker_enabled=true`: Retrieve `candidate_k` chunks, rerank, take `reranker_top_k`

### 3. Updated Schemas (`app/models/schemas.py`)

**QueryRequest additions:**
```python
reranker_enabled: bool = False
reranker_top_k: int = Field(default=5, ge=1, le=20)
candidate_k: int = Field(default=20, ge=1, le=50)
```

**Trace model additions:**
```python
reranker_enabled: Mapped[bool]
reranker_results: Mapped[dict]  # Stores reranked chunks with scores
```

**Latency breakdown now includes:**
```python
{
    "retrieval_ms": 120,
    "reranker_ms": 450,  # NEW
    "generation_ms": 700,
    "total_ms": 1270
}
```

### 4. Updated Tracer (`app/tracing/tracer.py`)

**New methods:**
- `start_reranker()` - Starts timing reranker phase
- `record_reranker(chunks)` - Records reranked results with scores and ranks

**Captured data per reranked chunk:**
- `chunk_id`, `text`, `document_name`, `page_number`
- `original_score` - Score from initial retrieval
- `reranker_score` - Score from cross-encoder
- `original_rank` - Position before reranking
- `reranker_rank` - Position after reranking

## API Changes

### Query Endpoint

**Request:**
```json
POST /query
{
  "question": "What is the refund policy?",
  "top_k": 5,
  "retrieval_mode": "hybrid",
  "reranker_enabled": true,
  "reranker_top_k": 5,
  "candidate_k": 20
}
```

**Response:**
```json
{
  "answer": "Customers can request refunds within 30 days.",
  "sources": [
    {
      "document": "refund_policy_2026.pdf",
      "page": 4,
      "chunk_id": "refund_2026_18",
      "similarity_score": 8.92  // reranker score when enabled
    }
  ],
  "latency_ms": 1270,
  "trace_id": "abc123-..."
}
```

### Trace Response

Now includes reranker details:
```json
{
  "reranker_enabled": true,
  "retrieval_results": {
    "chunks": [
      {
        "chunk_id": "...",
        "vector_score": 0.85,
        "bm25_score": 12.5,
        "original_rank": 3,
        ...
      }
    ]
  },
  "reranker_results": {
    "chunks": [
      {
        "chunk_id": "...",
        "original_score": 0.0328,
        "reranker_score": 8.92,
        "original_rank": 3,
        "reranker_rank": 1
      }
    ]
  },
  "latency": {
    "retrieval_ms": 120,
    "reranker_ms": 450,
    "generation_ms": 700,
    "total_ms": 1270
  }
}
```

## Technical Decisions

### Why Lazy Model Loading?
- Cross-encoder model is ~80MB
- Loading takes 2-5 seconds
- Not all queries use reranking
- Lazy loading keeps API responsive for non-reranked queries

### Why ms-marco-MiniLM-L-6-v2?
- **Fast**: 6 layers, ~22M parameters
- **Accurate**: Trained on MS MARCO dataset (real search queries)
- **Small**: Fits in memory, runs on CPU
- **Standard**: Widely used in production RAG systems

### Why candidate_k=20?
- Balance between accuracy and latency
- Too few: Might miss relevant chunks
- Too many: Reranking becomes slow
- 20 is a common sweet spot in practice

## Testing

### Test Coverage
- **test_reranker.py** (6 tests):
  - Basic reranking with score reordering
  - Top-k limiting
  - Empty chunks handling
  - Metadata preservation
  - BM25 result reranking
  - Hybrid result reranking

### Test Strategy
- Mock CrossEncoder to avoid loading actual model
- Test ranking logic independently
- Verify metadata flows through correctly
- Test all input chunk types (vector, BM25, hybrid)

## Files Created/Modified

### New Files
- `app/retrieval/reranker.py` - Cross-encoder reranker implementation
- `tests/test_reranker.py` - Unit tests for reranker
- `verify_milestone4.py` - Verification script

### Modified Files
- `app/models/schemas.py` - Added reranker config to QueryRequest and Trace
- `app/api/query.py` - Integrated reranker into query pipeline
- `app/tracing/tracer.py` - Added reranker timing and result capture
- `pyproject.toml` - Added sentence-transformers dependency
- `README.md` - Updated with reranker documentation

## Dependencies Added

```toml
"sentence-transformers>=2.3.0"  # Cross-encoder models
```

Note: This also pulls in torch and transformers (large dependencies).

## Performance Characteristics

### Latency Breakdown (typical)
- **Retrieval**: 100-200ms (vector/BM25/hybrid)
- **Reranking**: 300-800ms (depends on candidate_k)
- **Generation**: 500-1000ms (depends on answer length)
- **Total with reranking**: 900-2000ms
- **Total without reranking**: 600-1200ms

### Memory Usage
- **Cross-encoder model**: ~80MB
- **Loaded on first rerank query**
- **Stays in memory** for subsequent queries

## Usage Examples

### Without Reranking (Fast)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "retrieval_mode": "vector",
    "reranker_enabled": false
  }'
```

### With Reranking (Accurate)
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "retrieval_mode": "hybrid",
    "reranker_enabled": true,
    "reranker_top_k": 5,
    "candidate_k": 20
  }'
```

## Observability Benefits

With reranking enabled, traces now show:
1. **Rank changes**: Which chunks moved up/down after reranking
2. **Score comparison**: Original retrieval score vs reranker score
3. **Latency impact**: How much time reranking adds
4. **Effectiveness**: Whether reranking improved answer quality

This data is crucial for:
- Tuning retrieval parameters
- Deciding when to use reranking
- Debugging retrieval failures
- Comparing retrieval strategies

## Next Steps (Milestone 5)

With reranking in place, the next milestone will add:
- Retrieval evaluation metrics (Hit@K, MRR, NDCG)
- Benchmark dataset with labeled questions
- Automated evaluation pipeline
- Baseline vs improved strategy comparison

## Verification

Run the verification script:
```bash
cd backend
python3 verify_milestone4.py
```

Expected output:
```
============================================================
RAGScope Milestone 4 Verification
============================================================

Testing reranker module...
✓ Reranker module defined correctly
Testing schemas...
✓ Schemas updated correctly
Testing query endpoint...
✓ Query endpoint updated correctly
Testing tracer...
✓ Tracer updated correctly
Testing test files...
✓ Test files exist with required tests

============================================================
SUCCESS: All verification tests passed!
============================================================

Milestone 4 Complete:
  ✓ Cross-encoder reranker implemented
  ✓ QueryRequest supports reranker config
  ✓ Query pipeline: retrieve → rerank → generate
  ✓ Tracer captures reranker scores and timing
  ✓ Tests written
```

---

**Status**: ✓ Complete  
**Date**: August 5, 2026  
**Test Coverage**: 6 new tests, all passing  
**Dependencies**: sentence-transformers, torch, transformers
