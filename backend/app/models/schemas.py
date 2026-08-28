import uuid
from datetime import datetime

from pydantic import BaseModel, Field
from sqlalchemy import DateTime, Float, Integer, String, Text, func
from sqlalchemy.dialects.postgresql import JSON
from sqlalchemy.orm import Mapped, mapped_column

from app.database.session import Base


class Document(Base):
    __tablename__ = "documents"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    filename: Mapped[str] = mapped_column(String(255), nullable=False)
    file_path: Mapped[str] = mapped_column(String(500), nullable=False)
    num_pages: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    num_chunks: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="uploaded")
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    indexed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class Source(BaseModel):
    document: str
    page: int
    chunk_id: str
    similarity_score: float


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=1, max_length=1000)
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: str = Field(default="vector", pattern="^(vector|bm25|hybrid)$")
    reranker_enabled: bool = Field(default=False)
    reranker_top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=50)
    query_rewriting_enabled: bool = Field(default=False)
    query_rewriting_strategy: str = Field(default="llm", pattern="^(none|llm|hyde|decompose)$")
    corrective_rag_enabled: bool = Field(default=False)
    corrective_rag_max_retries: int = Field(default=2, ge=0, le=5)
    corrective_rag_relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    evaluation_enabled: bool = Field(default=False)
    grounding_check_enabled: bool = Field(default=False)
    grounding_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    latency_ms: int
    evaluation: dict | None = None
    grounding: dict | None = None


class DocumentResponse(BaseModel):
    id: str
    filename: str
    num_pages: int
    num_chunks: int
    status: str
    created_at: datetime
    indexed_at: datetime | None

    model_config = {"from_attributes": True}


class IndexResponse(BaseModel):
    document_id: str
    filename: str
    num_chunks: int
    status: str


class Trace(Base):
    __tablename__ = "traces"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    original_query: Mapped[str] = mapped_column(Text, nullable=False)
    rewritten_query: Mapped[str | None] = mapped_column(Text, nullable=True)
    retrieval_mode: Mapped[str] = mapped_column(String(20), nullable=False, default="vector")
    reranker_enabled: Mapped[bool] = mapped_column(default=False)
    query_rewriting_enabled: Mapped[bool] = mapped_column(default=False)
    query_rewriting_strategy: Mapped[str | None] = mapped_column(String(20), nullable=True)
    corrective_rag_enabled: Mapped[bool] = mapped_column(default=False)
    corrective_rag_retries: Mapped[int] = mapped_column(Integer, default=0)
    evaluation_enabled: Mapped[bool] = mapped_column(default=False)
    grounding_check_enabled: Mapped[bool] = mapped_column(default=False)
    retrieval_results: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    reranker_results: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    generation_result: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    evaluation_result: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    grounding_result: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    latency: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class RetrievedChunkTrace(BaseModel):
    chunk_id: str
    text: str
    document_name: str
    page_number: int
    vector_score: float | None = None
    bm25_score: float | None = None
    reranker_score: float | None = None
    final_rank: int | None = None


class LatencyBreakdown(BaseModel):
    retrieval_ms: int
    generation_ms: int
    total_ms: int


class GenerationTrace(BaseModel):
    answer: str
    citations: list[str]
    prompt_tokens: int | None = None
    completion_tokens: int | None = None


class TraceResponse(BaseModel):
    id: str
    original_query: str
    rewritten_query: str | None
    retrieval_mode: str
    reranker_enabled: bool
    query_rewriting_enabled: bool
    query_rewriting_strategy: str | None
    corrective_rag_enabled: bool
    corrective_rag_retries: int
    evaluation_enabled: bool
    grounding_check_enabled: bool
    retrieval_results: dict
    reranker_results: dict
    generation_result: dict
    evaluation_result: dict
    grounding_result: dict
    latency: dict
    created_at: datetime

    model_config = {"from_attributes": True}


class TraceListResponse(BaseModel):
    id: str
    original_query: str
    total_latency_ms: int
    num_sources: int
    created_at: datetime

    model_config = {"from_attributes": True}


class QueryResponseWithTrace(QueryResponse):
    trace_id: str


class Experiment(Base):
    __tablename__ = "experiments"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    name: Mapped[str] = mapped_column(String(255), nullable=False)
    description: Mapped[str | None] = mapped_column(Text, nullable=True)
    config: Mapped[dict] = mapped_column(JSON, nullable=False)
    status: Mapped[str] = mapped_column(String(50), nullable=False, default="created")
    dataset_path: Mapped[str] = mapped_column(String(500), nullable=False)
    total_queries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    completed_queries: Mapped[int] = mapped_column(Integer, nullable=False, default=0)
    avg_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_faithfulness: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_relevancy: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_precision: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_recall: Mapped[float | None] = mapped_column(Float, nullable=True)
    avg_grounding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    failure_analysis: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )
    completed_at: Mapped[datetime | None] = mapped_column(DateTime(timezone=True), nullable=True)


class ExperimentResult(Base):
    __tablename__ = "experiment_results"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=lambda: str(uuid.uuid4()))
    experiment_id: Mapped[str] = mapped_column(String(36), nullable=False)
    query_id: Mapped[str] = mapped_column(String(50), nullable=False)
    question: Mapped[str] = mapped_column(Text, nullable=False)
    expected_answer: Mapped[str] = mapped_column(Text, nullable=False)
    generated_answer: Mapped[str] = mapped_column(Text, nullable=False)
    expected_sources: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    retrieved_sources: Mapped[list] = mapped_column(JSON, nullable=False, default=list)
    latency_ms: Mapped[int] = mapped_column(Integer, nullable=False)
    faithfulness_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevancy_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    precision_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    grounding_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    is_grounded: Mapped[bool | None] = mapped_column(nullable=True)
    failure_type: Mapped[str | None] = mapped_column(String(255), nullable=True)
    failure_details: Mapped[dict] = mapped_column(JSON, nullable=False, default=dict)
    trace_id: Mapped[str | None] = mapped_column(String(36), nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True), server_default=func.now(), nullable=False
    )


class ExperimentConfig(BaseModel):
    """Configuration for an experiment."""
    top_k: int = Field(default=5, ge=1, le=20)
    retrieval_mode: str = Field(default="vector", pattern="^(vector|bm25|hybrid)$")
    reranker_enabled: bool = Field(default=False)
    reranker_top_k: int = Field(default=5, ge=1, le=20)
    candidate_k: int = Field(default=20, ge=1, le=50)
    query_rewriting_enabled: bool = Field(default=False)
    query_rewriting_strategy: str = Field(default="none", pattern="^(none|llm|hyde|decompose)$")
    corrective_rag_enabled: bool = Field(default=False)
    corrective_rag_max_retries: int = Field(default=2, ge=0, le=5)
    corrective_rag_relevance_threshold: float = Field(default=0.7, ge=0.0, le=1.0)
    evaluation_enabled: bool = Field(default=True)
    grounding_check_enabled: bool = Field(default=True)
    grounding_threshold: float = Field(default=0.7, ge=0.0, le=1.0)


class ExperimentCreateRequest(BaseModel):
    """Request to create a new experiment."""
    name: str = Field(..., min_length=1, max_length=255)
    description: str | None = Field(default=None, max_length=1000)
    config: ExperimentConfig
    dataset_path: str = Field(..., min_length=1)


class ExperimentResponse(BaseModel):
    """Response model for experiment data."""
    id: str
    name: str
    description: str | None
    config: dict
    status: str
    dataset_path: str
    total_queries: int
    completed_queries: int
    avg_latency_ms: float | None
    avg_faithfulness: float | None
    avg_relevancy: float | None
    avg_precision: float | None
    avg_recall: float | None
    avg_grounding_score: float | None
    failure_analysis: dict
    created_at: datetime
    completed_at: datetime | None

    model_config = {"from_attributes": True}


class ExperimentListResponse(BaseModel):
    """Response model for experiment list."""
    id: str
    name: str
    status: str
    total_queries: int
    completed_queries: int
    avg_latency_ms: float | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ExperimentResultResponse(BaseModel):
    """Response model for individual experiment result."""
    id: str
    experiment_id: str
    query_id: str
    question: str
    expected_answer: str
    generated_answer: str
    expected_sources: list
    retrieved_sources: list
    latency_ms: int
    faithfulness_score: float | None
    relevancy_score: float | None
    precision_score: float | None
    recall_score: float | None
    grounding_score: float | None
    is_grounded: bool | None
    failure_type: str | None
    failure_details: dict
    trace_id: str | None
    created_at: datetime

    model_config = {"from_attributes": True}


class ExperimentComparisonResponse(BaseModel):
    """Response model for comparing experiments."""
    experiment_ids: list[str]
    metrics: dict
    failures_by_type: dict
    recommendations: list[str]
