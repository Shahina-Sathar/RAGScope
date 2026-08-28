# Milestone 6: Query Rewriting + LangGraph Corrective RAG - Complete ✓

## Overview

Milestone 6 introduces intelligent query processing and self-correcting RAG workflows. The system can now automatically improve queries before retrieval and detect when retrieval quality is poor, triggering corrective actions to improve results.

## What Was Built

### 1. Query Rewriting Module (`app/query/rewriter.py`)

Implemented 4 query rewriting strategies:

**NoOpRewriter**
- Passthrough strategy that returns queries unchanged
- Useful as baseline or when rewriting is disabled

**LLMQueryRewriter**
- Uses LLM to rewrite queries for clarity and specificity
- Makes implicit references explicit
- Adds relevant context to vague queries
- Uses precise terminology
- Preserves original intent

**HyDERewriter (Hypothetical Document Embeddings)**
- Generates a hypothetical answer to the query
- Uses the hypothetical answer for retrieval
- Improves semantic matching by aligning query and document representations
- Particularly effective for complex or ambiguous queries

**QueryDecomposer**
- Breaks down complex queries into 2-3 simpler sub-queries
- Each sub-query is self-contained
- Enables retrieval of information from multiple angles
- Useful for multi-part questions

**Factory Function**
- `get_rewriter(strategy)` returns the appropriate rewriter
- Supports: "none", "llm", "hyde", "decompose"
- Raises ValueError for unknown strategies

### 2. LangGraph Corrective RAG (`app/query/corrective_rag.py`)

Implemented a self-correcting RAG workflow using LangGraph:

**CorrectiveRAGState**
- TypedDict defining workflow state
- Tracks: query, rewritten_query, documents, relevance_scores, is_relevant, retry_count, max_retries, final_answer

**CorrectiveRAG Workflow**
- **retrieve**: Retrieves documents for current query
- **grade_relevance**: Grades document relevance using similarity scores
- **rewrite_query**: Rewrites query using LLMQueryRewriter
- **generate_answer**: Generates final answer from documents

**Workflow Logic**
1. Start with original query
2. Retrieve documents
3. Grade relevance (avg similarity score >= threshold)
4. If not relevant and retries remain:
   - Rewrite query
   - Go back to step 2
5. If relevant or max retries reached:
   - Generate answer
   - End workflow

**Configuration**
- `max_retries`: Maximum number of retry attempts (default: 2)
- `relevance_threshold`: Minimum avg similarity score (default: 0.7)

### 3. Schema Updates (`app/models/schemas.py`)

**QueryRequest additions:**
```python
query_rewriting_enabled: bool = False
query_rewriting_strategy: str = "llm"  # none, llm, hyde, decompose
corrective_rag_enabled: bool = False
corrective_rag_max_retries: int = 2
corrective_rag_relevance_threshold: float = 0.7
```

**Trace model additions:**
```python
query_rewriting_enabled: Mapped[bool]
query_rewriting_strategy: Mapped[str | None]
corrective_rag_enabled: Mapped[bool]
corrective_rag_retries: Mapped[int]
```

### 4. Query API Integration (`app/api/query.py`)

Updated query endpoint to support new features:

**Flow:**
1. If `query_rewriting_enabled`, rewrite the query using selected strategy
2. If `corrective_rag_enabled`, use CorrectiveRAG workflow
3. Otherwise, use standard retrieval flow (with optional reranking)

**Integration Points:**
- Query rewriting happens before retrieval
- Corrective RAG replaces standard flow when enabled
- All actions are traced for observability

### 5. Tracer Updates (`app/tracing/tracer.py`)

Updated QueryTracer to capture new features:

**New fields:**
- `query_rewriting_enabled`: Whether rewriting was used
- `query_rewriting_strategy`: Which strategy was used
- `corrective_rag_enabled`: Whether corrective RAG was used
- `corrective_rag_retries`: Number of retries performed
- `rewritten_query`: The final rewritten query (if any)

## API Usage

### Query Rewriting

```bash
POST /query
{
  "question": "What is the refund policy?",
  "query_rewriting_enabled": true,
  "query_rewriting_strategy": "llm"
}
```

**Strategies:**
- `none`: No rewriting
- `llm`: LLM-based refinement
- `hyde`: Hypothetical document embeddings
- `decompose`: Query decomposition

### Corrective RAG

```bash
POST /query
{
  "question": "What is the refund policy?",
  "corrective_rag_enabled": true,
  "corrective_rag_max_retries": 2,
  "corrective_rag_relevance_threshold": 0.7
}
```

### Combined Usage

```bash
POST /query
{
  "question": "What is the refund policy?",
  "query_rewriting_enabled": true,
  "query_rewriting_strategy": "llm",
  "corrective_rag_enabled": true,
  "corrective_rag_max_retries": 2,
  "corrective_rag_relevance_threshold": 0.7,
  "retrieval_mode": "hybrid",
  "reranker_enabled": true
}
```

## Technical Decisions

### Why Multiple Rewriting Strategies?

Different queries benefit from different rewriting approaches:
- **LLM**: Best for vague or ambiguous queries
- **HyDE**: Best for complex semantic matching
- **Decompose**: Best for multi-part questions
- **None**: Baseline or when rewriting hurts performance

### Why LangGraph for Corrective RAG?

LangGraph provides:
- **State management**: Clean workflow state tracking
- **Conditional edges**: Easy retry logic
- **Visualization**: Graph-based workflow understanding
- **Extensibility**: Easy to add new nodes/edges

### Why Relevance Grading?

Not all retrievals are successful. Grading allows:
- **Detection**: Identify poor retrievals automatically
- **Correction**: Retry with improved queries
- **Observability**: Track how often corrections are needed
- **Tuning**: Adjust threshold based on use case

### Why Configurable Thresholds?

Different use cases have different quality requirements:
- **High threshold (0.8+)**: Strict quality, more retries
- **Medium threshold (0.6-0.8)**: Balanced approach
- **Low threshold (0.4-0.6)**: Lenient, fewer retries

## Files Created/Modified

### New Files
- `app/query/__init__.py`
- `app/query/rewriter.py` - Query rewriting strategies
- `app/query/corrective_rag.py` - LangGraph corrective RAG
- `tests/test_query_rewriter.py` - Tests for rewriting
- `tests/test_corrective_rag.py` - Tests for corrective RAG
- `backend/verify_milestone6.py` - Verification script

### Modified Files
- `pyproject.toml` - Added langgraph dependency
- `app/models/schemas.py` - Added new fields
- `app/api/query.py` - Integrated new features
- `app/tracing/tracer.py` - Capture new trace data
- `README.md` - Added Milestone 6 documentation

## Dependencies Added

```toml
"langgraph>=0.2.0"  # Workflow orchestration
```

## Test Coverage

**test_query_rewriter.py** (12 tests)
- NoOpRewriter tests
- LLMQueryRewriter tests
- HyDERewriter tests
- QueryDecomposer tests
- get_rewriter factory tests

**test_corrective_rag.py** (15+ tests)
- State management tests
- Retrieve node tests
- Grade relevance tests
- Should retry logic tests
- Rewrite query tests
- Generate answer tests
- Full workflow tests

## Observability Benefits

With these features, traces now show:
1. **Query transformations**: Original vs rewritten queries
2. **Rewriting strategy**: Which strategy was used
3. **Corrective actions**: Whether corrective RAG was triggered
4. **Retry count**: How many retries were needed
5. **Effectiveness**: Whether corrections improved results

This data is crucial for:
- Tuning rewriting strategies
- Adjusting relevance thresholds
- Understanding when corrections are needed
- Optimizing the overall pipeline

## Performance Considerations

### Query Rewriting
- **Latency**: 200-500ms per rewrite (LLM call)
- **Trade-off**: Improved retrieval vs added latency
- **Best for**: Complex or ambiguous queries

### Corrective RAG
- **Latency**: 500-1500ms per retry cycle
- **Max latency**: max_retries * cycle_time
- **Trade-off**: Better results vs higher latency
- **Best for**: Critical queries where accuracy matters

## Usage Recommendations

### When to Use Query Rewriting
- Vague or ambiguous queries
- Queries with implicit context
- Multi-part questions (use decompose)
- When baseline retrieval is poor

### When to Use Corrective RAG
- Critical queries requiring high accuracy
- When retrieval quality is inconsistent
- When you can tolerate higher latency
- For production systems with SLAs

### When NOT to Use
- Simple, clear queries (rewriting may hurt)
- Latency-sensitive applications
- When baseline retrieval is already good
- During development/testing (adds complexity)

## Next Steps (Milestone 7)

With intelligent query processing in place, the next milestone will add:
- RAGAS evaluation framework
- Grounding/hallucination detection
- Faithfulness metrics
- Answer relevance scoring
- Context precision/recall metrics

## Verification

Run the verification script:
```bash
cd backend
python3 verify_milestone6.py
```

Expected output:
```
======================================================================
RAGScope Milestone 6 Verification
Query Rewriting + LangGraph Corrective RAG
======================================================================

Testing query rewriter module...
✓ Query rewriter module verified
Testing corrective RAG module...
✓ corrective RAG module verified
Testing langgraph dependency...
✓ LangGraph dependency verified
Testing schema updates...
✓ Schema updates verified
Testing query API integration...
✓ Query API integration verified
Testing tracer updates...
✓ Tracer updates verified
Testing test files...
✓ Test files verified

======================================================================
SUCCESS: All verification tests passed!
======================================================================

Milestone 6 Complete:
  ✓ Query rewriting module with 4 strategies (none, llm, hyde, decompose)
  ✓ LangGraph corrective RAG workflow
  ✓ Schema updates for new configuration options
  ✓ Query API integration
  ✓ Tracer updates for tracking new features
  ✓ Test files created
```

---

**Status**: ✓ Complete  
**Date**: August 5, 2026  
**Test Coverage**: 27+ tests  
**Dependencies**: langgraph
