# RAGScope

RAG Evaluation & Observability Platform - A tool for debugging, experimenting with, and evaluating RAG pipelines.

## Overview

RAGScope is not a standard document chatbot. It's a RAG debugging and experimentation platform that allows developers to:

- Upload and parse PDF documents
- Generate embeddings and store in Pinecone
- Ask questions and retrieve relevant chunks
- Generate grounded answers with citations
- Inspect retrieval metadata and scoring
- Evaluate RAG pipeline performance

## Tech Stack

- **Python 3.12**
- **FastAPI** - Web framework
- **Pinecone** - Vector database
- **LangChain** - LLM orchestration
- **PostgreSQL** - Application metadata
- **Pydantic v2** - Data validation
- **pytest** - Testing

## Project Structure

```
ragscope/
├── backend/
│   ├── app/
│   │   ├── api/              # FastAPI endpoints
│   │   ├── config/           # Configuration management
│   │   ├── database/         # Database setup
│   │   ├── generation/       # LLM answer generation
│   │   ├── ingestion/        # PDF parsing, chunking, indexing
│   │   ├── models/           # Data models and schemas
│   │   └── retrieval/        # Vector search
│   └── pyproject.toml
├── tests/                    # Test suite
├── data/                     # Uploaded documents
└── docker-compose.yml        # PostgreSQL setup
```

## Setup

### Prerequisites

- Python 3.12+
- UV package manager
- Docker (for PostgreSQL)
- OpenAI API key
- Pinecone API key

### Installation

1. **Clone and navigate to the project:**
   ```bash
   cd RAGScope
   ```

2. **Install dependencies:**
   ```bash
   cd backend
   uv sync
   ```

3. **Set up environment variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

   Required environment variables:
   - `OPENAI_API_KEY` - Your OpenAI API key
   - `PINECONE_API_KEY` - Your Pinecone API key
   - `PINECONE_HOST` - Your Pinecone index host
   - `PINECONE_INDEX_NAME` - Name of your Pinecone index (default: ragscope)

4. **Start PostgreSQL:**
   ```bash
   docker-compose up -d postgres
   ```

5. **Run the application:**
   ```bash
   uv run uvicorn app.main:app --reload
   ```

   The API will be available at `http://localhost:8000`

## API Endpoints

### Document Management

**Upload PDF:**
```bash
POST /documents/upload
Content-Type: multipart/form-data

curl -X POST http://localhost:8000/documents/upload \
  -F "file=@document.pdf"
```

**Index Document:**
```bash
POST /documents/index/{document_id}

curl -X POST http://localhost:8000/documents/index/{document_id}
```

### Query

**Ask a question:**
```bash
POST /query

curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the refund policy?",
    "top_k": 5,
    "retrieval_mode": "vector",
    "reranker_enabled": false,
    "reranker_top_k": 5,
    "candidate_k": 20
  }'
```

**Retrieval Modes:**
- `vector` (default) - Semantic similarity search using embeddings
- `bm25` - Lexical keyword matching using BM25 algorithm
- `hybrid` - Combines vector + BM25 using Reciprocal Rank Fusion (RRF)

**Reranker Configuration:**
- `reranker_enabled` - Enable cross-encoder reranking (default: false)
- `reranker_top_k` - Number of chunks to return after reranking (default: 5)
- `candidate_k` - Number of candidates to retrieve before reranking (default: 20)

**Pipeline with Reranking:**
```
Query → Retrieve (candidate_k) → Rerank → Top (reranker_top_k) → Generate
```

Response:
```json
{
  "answer": "Customers can request refunds within 30 days.",
  "sources": [
    {
      "document": "refund_policy_2026.pdf",
      "page": 4,
      "chunk_id": "refund_2026_18",
      "similarity_score": 0.91
    }
  ],
  "latency_ms": 820,
  "trace_id": "abc123-def456-..."
}
```

### Traces

**List recent traces:**
```bash
GET /traces?limit=20&offset=0
```

**Get trace details:**
```bash
GET /traces/{trace_id}
```

Response:
```json
{
  "id": "abc123-...",
  "original_query": "What is the refund policy?",
  "rewritten_query": null,
  "retrieval_mode": "vector",
  "retrieval_results": {
    "chunks": [
      {
        "chunk_id": "refund_2026_18",
        "text": "...",
        "document_name": "refund_policy_2026.pdf",
        "page_number": 4,
        "vector_score": 0.91,
        "bm25_score": null,
        "reranker_score": null,
        "final_rank": 1
      }
    ]
  },
  "generation_result": {
    "answer": "Customers can request refunds within 30 days.",
    "citations": ["refund_2026_18"],
    "prompt_tokens": 450,
    "completion_tokens": 120
  },
  "latency": {
    "retrieval_ms": 120,
    "generation_ms": 700,
    "total_ms": 820
  },
  "created_at": "2026-08-05T10:30:00Z"
}
```

### Health Check

```bash
GET /health
```

## Testing

Run the test suite:
```bash
uv run pytest tests/ -v
```

Run with coverage:
```bash
uv run pytest tests/ --cov=app --cov-report=html
```

## Linting

```bash
uv run ruff check app/ tests/
```

## Pinecone Setup

1. Create a Pinecone account at https://www.pinecone.io/
2. Create an index with the following configuration:
   - **Name:** ragscope
   - **Dimension:** 1536 (for text-embedding-3-small)
   - **Metric:** cosine
3. Copy your API key and host to `.env`

## Development

### Running in development mode:
```bash
uv run uvicorn app.main:app --reload --port 8000
```

### Viewing API documentation:
- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## Current Status

**Milestone 1: Baseline RAG** ✓ Complete

Implemented:
- PDF parsing with page number preservation
- Text chunking with configurable size/overlap
- Embedding generation using OpenAI
- Vector storage in Pinecone
- Vector similarity search
- LLM-based answer generation with citations
- Complete API endpoints
- Test suite

**Milestone 2: Retrieval Observability** ✓ Complete

Implemented:
- Query tracing system that captures complete RAG pipeline execution
- Trace storage in PostgreSQL with full retrieval metadata
- Trace retrieval endpoints for inspection
- Latency breakdown (retrieval vs generation)
- Citation tracking and source attribution

**Milestone 3: BM25 + Hybrid Retrieval** ✓ Complete

Implemented:
- BM25 lexical retrieval for keyword matching
- Hybrid retrieval combining vector + BM25
- Reciprocal Rank Fusion (RRF) for score combination
- Three retrieval modes: vector, bm25, hybrid
- Automatic BM25 index building during document indexing
- Case-insensitive BM25 matching

**Milestone 4: Reranking** ✓ Complete

Implemented:
- Cross-encoder reranker using sentence-transformers
- Configurable reranking pipeline: retrieve → rerank → generate
- Preserves original retrieval scores alongside reranker scores
- Lazy model loading for faster startup
- Full reranker tracing with timing and score comparison

**Milestone 5: Retrieval Metrics and Benchmark Dataset** ✓ Complete

Implemented:
- Retrieval evaluation metrics: Hit@K, Recall@K, Precision@K, MRR, MAP@K
- Benchmark dataset with 20 questions across 7 categories
- Evaluation runner for executing benchmarks and calculating metrics
- Comprehensive test suite (41 tests)
- Support for per-category metric analysis

**Milestone 6: Query Rewriting + LangGraph Corrective RAG** ✓ Complete

Implemented:
- Query rewriting module with 4 strategies:
  - `none`: No rewriting (passthrough)
  - `llm`: LLM-based query refinement for clarity and specificity
  - `hyde`: Hypothetical Document Embeddings for improved semantic matching
  - `decompose`: Complex query decomposition into sub-queries
- LangGraph-based corrective RAG workflow:
  - Automatic relevance grading of retrieved documents
  - Query rewriting when documents are not relevant
  - Configurable retry mechanism with max retries limit
  - Relevance threshold tuning
- Full integration with query API and tracing system
- Traces capture rewritten queries and corrective actions

**Milestone 7: RAGAS + Grounding Evaluation** ✓ Complete

Implemented:
- RAGAS evaluation framework integration:
  - Faithfulness: Measures factual consistency with context
  - Answer relevancy: Measures how well answer addresses question
  - Context precision: Measures relevance of retrieved context
  - Context recall: Measures completeness of retrieved context
- Grounding and hallucination detection:
  - LLM-based grounding checker for claim verification
  - Simple keyword-based grounding checker (fast alternative)
  - Hallucination detection with supported/hallucinated claims
  - Configurable grounding threshold
- Full integration with query pipeline:
  - Optional evaluation and grounding checks per query
  - Evaluation results included in query responses
  - Evaluation and grounding results persisted in traces
  - Timing metrics for evaluation overhead
- Comprehensive test suite for evaluation modules

**Milestone 8: Experiment Framework + Failure Analyzer** ✓ Complete

Implemented:
- Experiment framework for systematic RAG testing:
  - Experiment configuration management with all RAG parameters
  - Experiment runner with evaluation dataset support
  - Background execution for long-running experiments
  - Experiment status tracking (created, running, completed, failed)
  - Result storage with full query details and metrics
- Failure analyzer with pattern detection:
  - Automatic failure type classification (retrieval, grounding, faithfulness, etc.)
  - Compound failure detection (multiple failures per query)
  - Failure rate calculation and statistics
  - Actionable recommendations for each failure type
  - Failure summary and trend analysis
- Experiment comparison and reporting:
  - Side-by-side comparison of multiple experiments
  - Metrics comparison (latency, faithfulness, relevancy, precision, recall, grounding)
  - Failure pattern comparison across experiments
  - Automated recommendations based on comparison results
  - Best experiment identification per metric
- Complete API endpoints:
  - Create, list, get, delete experiments
  - Run experiments in background
  - Get experiment results with filtering
  - Analyze failures and get summaries
  - Compare multiple experiments
- Comprehensive test coverage for all components

**Milestone 9: Streamlit Dashboard** ✓ Complete

Implemented:
- Complete Streamlit dashboard with 6 interactive pages:
  - **Home**: System overview, metrics, recent activity
  - **Documents**: PDF upload, indexing, management
  - **Query Playground**: Interactive query testing with all RAG options
  - **Traces**: Detailed trace inspection with scores and metadata
  - **Experiments**: Experiment creation, execution, and result viewing
  - **Comparison**: Visual comparison with Plotly charts (bar, radar, failure patterns)
- API client for backend communication
- Environment-aware configuration (Docker and local development)
- Real-time query execution with immediate feedback
- Interactive visualizations for experiment comparison
- Status indicators and error handling throughout

**Milestone 10: Docker, Testing, Documentation** ✓ Complete

Implemented:
- Docker containerization:
  - Backend Dockerfile (Python 3.12, UV, FastAPI)
  - Frontend Dockerfile (Python 3.12, UV, Streamlit)
  - Docker Compose with PostgreSQL, backend, and frontend services
  - Health checks and proper networking
  - Volume mounts for data persistence
- Comprehensive test suite:
  - 150+ unit tests across all modules
  - Integration tests for API endpoints
  - Verification scripts for each milestone
- Complete documentation:
  - README with setup instructions
  - Milestone summaries (MILESTONE_1_SUMMARY.md through MILESTONE_10_SUMMARY.md)
  - API documentation (auto-generated from FastAPI)
  - Docker deployment guide

## Quick Start

### Local Development

1. **Start PostgreSQL:**
   ```bash
   docker-compose up -d postgres
   ```

2. **Start Backend:**
   ```bash
   cd backend
   uv sync
   uv run uvicorn app.main:app --reload
   ```

3. **Start Frontend:**
   ```bash
   cd frontend
   uv sync
   uv run streamlit run app.py
   ```

4. **Access:**
   - Dashboard: http://localhost:8501
   - API: http://localhost:8000
   - API Docs: http://localhost:8000/docs

### Docker Deployment

```bash
# Build and start all services
docker-compose up --build

# Access services
# Dashboard: http://localhost:8501
# API: http://localhost:8000
```

## Project Structure

```
ragscope/
├── backend/                    # FastAPI backend
│   ├── app/
│   │   ├── api/               # API endpoints
│   │   ├── config/            # Configuration
│   │   ├── database/          # Database setup
│   │   ├── evaluation/        # RAGAS, metrics, grounding
│   │   ├── experiments/       # Experiment framework
│   │   ├── generation/        # LLM generation
│   │   ├── ingestion/         # PDF parsing, chunking, indexing
│   │   ├── models/            # Data models
│   │   ├── query/             # Query rewriting, corrective RAG
│   │   ├── retrieval/         # Vector, BM25, hybrid, reranking
│   │   └── tracing/           # Query tracing
│   ├── tests/                 # Test suite
│   └── pyproject.toml         # Backend dependencies
├── frontend/                   # Streamlit dashboard
│   ├── pages/                 # Dashboard pages
│   ├── utils/                 # API client
│   ├── app.py                 # Main application
│   └── pyproject.toml         # Frontend dependencies
├── evaluation/                 # Benchmark datasets
├── docker-compose.yml          # Docker orchestration
├── Dockerfile.backend          # Backend container
├── Dockerfile.frontend         # Frontend container
└── README.md                   # This file
```

## Testing

Run all tests:
```bash
cd backend
uv run pytest tests/ -v
```

Run verification scripts:
```bash
cd backend
for i in {1..10}; do python3 verify_milestone$i.py; done
```

## License

MIT
