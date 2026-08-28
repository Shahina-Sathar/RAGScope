# Milestone 8: Experiment Framework + Failure Analyzer - Complete ✓

## Overview

Milestone 8 adds a comprehensive experiment framework for systematic RAG testing and a failure analyzer for identifying and understanding RAG pipeline issues. This enables data-driven optimization of RAG configurations through automated experimentation and failure analysis.

## What Was Built

### 1. Experiment Models and Database Schema (`app/models/schemas.py`)

**Experiment Model**
- Stores experiment metadata and configuration
- Tracks status (created, running, completed, failed)
- Stores aggregate metrics (avg latency, faithfulness, relevancy, precision, recall, grounding)
- Records failure analysis results
- Timestamps for creation and completion

**ExperimentResult Model**
- Stores individual query results for each experiment
- Captures question, expected answer, generated answer
- Stores expected and retrieved sources
- Records all evaluation metrics per query
- Tracks failure type and details for each query
- Links to trace for full observability

**Pydantic Models**
- `ExperimentConfig`: Configuration for experiments (all RAG parameters)
- `ExperimentCreateRequest`: Request to create new experiment
- `ExperimentResponse`: Full experiment data with metrics
- `ExperimentListResponse`: Summary for experiment lists
- `ExperimentResultResponse`: Individual query result data
- `ExperimentComparisonResponse`: Comparison results with recommendations

### 2. Experiment Runner (`app/experiments/runner.py`)

**ExperimentRunner Class**
- Executes experiments with specific configurations
- Loads evaluation datasets and runs all queries
- Integrates with retrieval, reranking, generation, evaluation, and grounding
- Calculates aggregate metrics across all queries
- Updates experiment status and metrics in database
- Handles errors gracefully and marks experiments as failed

**Key Features**
- Background execution support for long-running experiments
- Progress tracking (completed_queries counter)
- Automatic metric aggregation
- Error handling and status updates
- Integration with all RAG components

### 3. Failure Analyzer (`app/experiments/analyzer.py`)

**FailureAnalyzer Class**
- Analyzes experiment results to identify failure patterns
- Classifies failures into specific types:
  - `RETRIEVAL_FAILURE`: No documents retrieved
  - `LOW_RETRIEVAL_QUALITY`: Low retrieval scores
  - `GROUNDING_FAILURE`: Answer not grounded in context
  - `FAITHFULNESS_FAILURE`: Low faithfulness score
  - `RELEVANCY_FAILURE`: Low relevancy score
  - `PRECISION_FAILURE`: Low precision score
  - `RECALL_FAILURE`: Low recall score
  - `HIGH_LATENCY`: Query latency too high
- Detects compound failures (multiple failures per query)
- Provides actionable recommendations for each failure type
- Calculates failure rates and statistics
- Generates failure summaries

**Key Features**
- Pattern detection across multiple queries
- Compound failure analysis
- Failure type counting and ranking
- Recommendation generation
- Integration with experiment results

### 4. Experiment Comparator (`app/experiments/comparison.py`)

**ExperimentComparator Class**
- Compares multiple experiments side-by-side
- Compares metrics across experiments:
  - Latency (min, max, avg, best)
  - Faithfulness (min, max, avg, best)
  - Relevancy (min, max, avg, best)
  - Precision (min, max, avg, best)
  - Recall (min, max, avg, best)
  - Grounding (min, max, avg, best)
- Compares failure patterns across experiments
- Generates automated recommendations based on comparison
- Identifies best experiment per metric

**Key Features**
- Multi-experiment comparison
- Statistical analysis (min, max, avg)
- Best experiment identification
- Failure pattern comparison
- Automated recommendations

### 5. Experiment API Endpoints (`app/api/experiments.py`)

**Endpoints**
- `POST /experiments` - Create new experiment
- `GET /experiments` - List all experiments (with filtering)
- `GET /experiments/{id}` - Get specific experiment
- `POST /experiments/{id}/run` - Run experiment in background
- `GET /experiments/{id}/results` - Get experiment results (with filtering)
- `POST /experiments/{id}/analyze` - Analyze failures
- `GET /experiments/{id}/failure-summary` - Get failure summary
- `POST /experiments/compare` - Compare multiple experiments
- `DELETE /experiments/{id}` - Delete experiment

**Features**
- Background execution for long-running experiments
- Filtering by status and failure type
- Pagination support
- Error handling and validation
- Integration with all experiment components

### 6. Test Coverage

**test_experiment_runner.py** (10 tests)
- Experiment not found handling
- Status updates during execution
- Dataset error handling
- Query execution with evaluation
- Query execution without evaluation
- Reranking integration
- Metric calculation

**test_failure_analyzer.py** (12 tests)
- No results handling
- No failures detection
- Retrieval failure detection
- Grounding failure detection
- Low metrics detection
- High latency detection
- Multiple failure types
- Compound failure detection
- Failure summary generation

**test_experiment_comparator.py** (10 tests)
- Insufficient experiments handling
- Experiments not found handling
- Successful comparison
- Metrics comparison
- Failures comparison
- Recommendation generation (faithfulness, latency, failures, grounding)

## API Usage Examples

### Create Experiment

```bash
POST /experiments
{
  "name": "Baseline Vector Search",
  "description": "Test baseline vector search configuration",
  "config": {
    "top_k": 5,
    "retrieval_mode": "vector",
    "reranker_enabled": false,
    "evaluation_enabled": true,
    "grounding_check_enabled": true
  },
  "dataset_path": "evaluation/questions.json"
}
```

### Run Experiment

```bash
POST /experiments/{id}/run
```

### Get Experiment Results

```bash
GET /experiments/{id}/results?failure_type=RETRIEVAL_FAILURE
```

### Analyze Failures

```bash
POST /experiments/{id}/analyze
```

Response:
```json
{
  "total_queries": 20,
  "total_failures": 5,
  "failure_rate": 0.25,
  "failure_types": {
    "RETRIEVAL_FAILURE": 2,
    "GROUNDING_FAILURE": 2,
    "HIGH_LATENCY": 1
  },
  "failures": [
    {
      "query_id": "Q001",
      "question": "What is the refund policy?",
      "failure_type": "RETRIEVAL_FAILURE",
      "details": {"reason": "No documents retrieved"},
      "recommendation": "Check if documents are properly indexed or adjust retrieval parameters"
    }
  ]
}
```

### Compare Experiments

```bash
POST /experiments/compare
{
  "experiment_ids": ["exp-1", "exp-2", "exp-3"]
}
```

Response:
```json
{
  "experiment_ids": ["exp-1", "exp-2", "exp-3"],
  "experiments": [...],
  "metrics_comparison": {
    "faithfulness": {
      "min": 0.75,
      "max": 0.90,
      "avg": 0.83,
      "best_experiment": "exp-3"
    }
  },
  "failures_comparison": {
    "exp-1": {"total_queries": 20, "queries_with_failures": 5, "failure_counts": {...}},
    "exp-2": {"total_queries": 20, "queries_with_failures": 3, "failure_counts": {...}},
    "exp-3": {"total_queries": 20, "queries_with_failures": 1, "failure_counts": {...}}
  },
  "recommendations": [
    "Experiment 'Full Pipeline' has the best faithfulness score. Consider using its configuration",
    "Most common failure type is 'RETRIEVAL_FAILURE'. Focus on addressing this issue"
  ]
}
```

## Files Created/Modified

### New Files
- `app/experiments/__init__.py` - Module initialization
- `app/experiments/runner.py` - Experiment runner
- `app/experiments/analyzer.py` - Failure analyzer
- `app/experiments/comparison.py` - Experiment comparator
- `app/api/experiments.py` - API endpoints
- `tests/test_experiment_runner.py` - Runner tests
- `tests/test_failure_analyzer.py` - Analyzer tests
- `tests/test_experiment_comparator.py` - Comparator tests
- `backend/verify_milestone8.py` - Verification script

### Modified Files
- `app/models/schemas.py` - Added experiment models
- `app/main.py` - Integrated experiments router
- `README.md` - Added Milestone 8 documentation

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Experiment API                        │
│  (Create, Run, List, Get, Compare, Analyze, Delete)     │
└────────────────┬────────────────────────────────────────┘
                 │
    ┌────────────┼────────────┬──────────────┐
    │            │            │              │
    ▼            ▼            ▼              ▼
┌─────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐
│ Runner  │  │ Analyzer │  │ Comparator│  │ Results  │
│         │  │          │  │          │  │ Storage  │
│ Execute │  │ Detect   │  │ Compare  │  │          │
│ queries │  │ failures │  │ metrics  │  │ Store    │
│         │  │          │  │          │  │ results  │
└────┬────┘  └────┬─────┘  └────┬─────┘  └──────────┘
     │            │              │
     └────────────┴──────────────┘
                 │
                 ▼
        ┌────────────────┐
        │  RAG Pipeline  │
        │                │
        │  Retrieval     │
        │  Reranking     │
        │  Generation    │
        │  Evaluation    │
        │  Grounding     │
        └────────────────┘
```

## Usage Workflow

1. **Create Experiment**
   - Define configuration (retrieval mode, top_k, reranking, etc.)
   - Specify evaluation dataset
   - Experiment created with status "created"

2. **Run Experiment**
   - Start background execution
   - Status changes to "running"
   - Each query executed with full RAG pipeline
   - Results stored in database
   - Progress tracked (completed_queries)

3. **Analyze Failures**
   - After completion, analyze results
   - Failure types detected and classified
   - Recommendations generated
   - Failure summary calculated

4. **Compare Experiments**
   - Select multiple experiments
   - Compare metrics side-by-side
   - Identify best experiment per metric
   - Get automated recommendations

5. **Iterate and Optimize**
   - Use recommendations to improve configuration
   - Create new experiment with improved config
   - Compare results
   - Repeat until satisfied

## Benefits

1. **Systematic Testing**
   - Test different configurations systematically
   - Compare results objectively
   - Track improvements over time

2. **Failure Understanding**
   - Automatically detect failure patterns
   - Understand why RAG fails
   - Get actionable recommendations

3. **Data-Driven Optimization**
   - Make decisions based on metrics
   - Identify best configurations
   - Optimize for specific metrics

4. **Reproducibility**
   - All experiments stored in database
   - Full traceability
   - Easy to reproduce results

5. **Scalability**
   - Background execution
   - Handle large datasets
   - Compare many experiments

## Next Steps (Milestone 9)

With the experiment framework in place, the next milestone will add:
- Streamlit dashboard for visualization
- Interactive experiment management
- Real-time experiment monitoring
- Visual comparison charts
- Failure analysis visualization
- Configuration management UI

## Verification

Run the verification script:
```bash
cd backend
python3 verify_milestone8.py
```

Expected output:
```
======================================================================
Milestone 8 Verification: Experiment Framework + Failure Analyzer
======================================================================

1. Checking experiment models...
   ✓ All experiment models present

2. Checking experiment runner...
   ✓ Experiment runner implemented

3. Checking failure analyzer...
   ✓ Failure analyzer implemented

4. Checking experiment comparator...
   ✓ Experiment comparator implemented

5. Checking experiment API endpoints...
   ✓ All API endpoints implemented

6. Checking main.py integration...
   ✓ Experiments router integrated in main.py

7. Checking test files...
   ✓ All test files present

8. Checking experiments module...
   ✓ Experiments module initialized

======================================================================
✅ VERIFICATION PASSED
======================================================================

Milestone 8 is complete!

Key features implemented:
  • Experiment models and database schema
  • Experiment configuration management
  • Experiment runner with evaluation support
  • Failure analyzer with pattern detection
  • Experiment comparison and recommendations
  • Complete API endpoints
  • Comprehensive test coverage
```

---

**Status**: ✓ Complete  
**Date**: August 5, 2026  
**Test Coverage**: 32 tests  
**Components**: 4 main modules, 9 API endpoints, 3 test files
