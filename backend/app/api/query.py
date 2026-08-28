import time

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.database.session import get_session
from app.models.schemas import QueryRequest, QueryResponseWithTrace, Source
from app.retrieval.factory import RetrievalFactory, RetrievalMode
from app.retrieval.vector import VectorRetriever, RetrievedChunk
from app.retrieval.bm25 import BM25Result
from app.retrieval.hybrid import HybridChunk
from app.retrieval.reranker import CrossEncoderReranker, RerankedChunk
from app.generation.generator import AnswerGenerator
from app.tracing.tracer import QueryTracer
from app.query.rewriter import get_rewriter
from app.query.corrective_rag import CorrectiveRAG
from app.evaluation.ragas_eval import RAGASEvaluator
from app.evaluation.grounding import GroundingChecker

router = APIRouter(tags=["query"])


@router.post("/query", response_model=QueryResponseWithTrace)
async def query(
    request: QueryRequest,
    session: AsyncSession = Depends(get_session),
):
    tracer = QueryTracer(
        original_query=request.question,
        retrieval_mode=request.retrieval_mode,
        reranker_enabled=request.reranker_enabled,
        query_rewriting_enabled=request.query_rewriting_enabled,
        query_rewriting_strategy=request.query_rewriting_strategy if request.query_rewriting_enabled else None,
        corrective_rag_enabled=request.corrective_rag_enabled,
        evaluation_enabled=request.evaluation_enabled,
        grounding_check_enabled=request.grounding_check_enabled,
    )

    # Step 1: Query rewriting (if enabled)
    query_text = request.question
    if request.query_rewriting_enabled and request.query_rewriting_strategy != "none":
        rewriter = get_rewriter(request.query_rewriting_strategy)
        query_text = rewriter.rewrite(request.question)
        tracer.rewritten_query = query_text

    # Step 2: Use corrective RAG if enabled
    if request.corrective_rag_enabled:
        retriever = VectorRetriever()
        corrective_rag = CorrectiveRAG(
            retriever=retriever,
            max_retries=request.corrective_rag_max_retries,
            relevance_threshold=request.corrective_rag_relevance_threshold,
        )
        
        tracer.start_retrieval()
        result = corrective_rag.run(query_text)
        tracer.record_retrieval(result["documents"])
        tracer.corrective_rag_retries = result["retry_count"]
        
        # Update rewritten query if corrective RAG rewrote it
        if result["rewritten_query"] and not tracer.rewritten_query:
            tracer.rewritten_query = result["rewritten_query"]
        
        # Use the final answer from corrective RAG
        answer = result["final_answer"]
        final_chunks = result["documents"]
        
        # Skip reranking and generation for corrective RAG
        tracer.start_generation()
        tracer.record_generation(answer=answer, citations=[])
        
    else:
        # Step 3: Standard retrieval flow
        tracer.start_retrieval()
        factory = RetrievalFactory()
        mode = RetrievalMode(request.retrieval_mode)
        
        retrieve_k = request.candidate_k if request.reranker_enabled else request.top_k
        chunks, mode_name = factory.retrieve(query_text, mode, top_k=retrieve_k)
        tracer.record_retrieval(chunks)

        final_chunks = chunks
        if request.reranker_enabled and len(chunks) > 0:
            tracer.start_reranker()
            reranker = CrossEncoderReranker()
            reranked = reranker.rerank(
                query_text, chunks, top_k=request.reranker_top_k
            )
            tracer.record_reranker(reranked)
            final_chunks = reranked

        tracer.start_generation()
        generator = AnswerGenerator()
        
        retrieved_chunks = []
        for chunk in final_chunks:
            if isinstance(chunk, RetrievedChunk):
                retrieved_chunks.append(chunk)
            elif isinstance(chunk, RerankedChunk):
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        similarity_score=chunk.reranker_score,
                    )
                )
            elif isinstance(chunk, (BM25Result, HybridChunk)):
                retrieved_chunks.append(
                    RetrievedChunk(
                        chunk_id=chunk.chunk_id,
                        text=chunk.text,
                        document_name=chunk.document_name,
                        page_number=chunk.page_number,
                        similarity_score=getattr(chunk, "bm25_score", 0.0) or 0.0,
                    )
                )
        
        answer, cited_chunk_ids = generator.generate(query_text, retrieved_chunks)
        tracer.record_generation(answer=answer, citations=cited_chunk_ids)

    # Step 4: Evaluation (if enabled)
    evaluation_data = None
    if request.evaluation_enabled:
        tracer.start_evaluation()
        try:
            evaluator = RAGASEvaluator()
            contexts = [chunk.text for chunk in final_chunks]
            eval_result = await evaluator.evaluate(
                question=request.question,
                answer=answer,
                contexts=contexts,
            )
            tracer.record_evaluation(
                faithfulness_score=eval_result.faithfulness_score,
                answer_relevancy_score=eval_result.answer_relevancy_score,
                context_precision_score=eval_result.context_precision_score,
                context_recall_score=eval_result.context_recall_score,
                error=eval_result.error,
            )
            evaluation_data = {
                "faithfulness_score": eval_result.faithfulness_score,
                "answer_relevancy_score": eval_result.answer_relevancy_score,
                "context_precision_score": eval_result.context_precision_score,
                "context_recall_score": eval_result.context_recall_score,
                "error": eval_result.error,
            }
        except Exception as e:
            tracer.record_evaluation(error=str(e))
            evaluation_data = {"error": str(e)}

    # Step 5: Grounding check (if enabled)
    grounding_data = None
    if request.grounding_check_enabled:
        try:
            grounding_checker = GroundingChecker()
            contexts = [chunk.text for chunk in final_chunks]
            grounding_result = grounding_checker.check_grounding(
                question=request.question,
                answer=answer,
                contexts=contexts,
            )
            tracer.record_grounding(
                is_grounded=grounding_result.is_grounded,
                grounding_score=grounding_result.grounding_score,
                supported_claims=grounding_result.supported_claims,
                hallucinated_claims=grounding_result.hallucinated_claims,
                explanation=grounding_result.explanation,
            )
            grounding_data = {
                "is_grounded": grounding_result.is_grounded,
                "grounding_score": grounding_result.grounding_score,
                "supported_claims": grounding_result.supported_claims,
                "hallucinated_claims": grounding_result.hallucinated_claims,
                "explanation": grounding_result.explanation,
            }
        except Exception as e:
            grounding_data = {"error": str(e)}

    trace_id = await tracer.persist(session)

    sources = []
    for chunk in final_chunks:
        score = 0.0
        if isinstance(chunk, RetrievedChunk):
            score = chunk.similarity_score
        elif isinstance(chunk, RerankedChunk):
            score = chunk.reranker_score
        elif isinstance(chunk, BM25Result):
            score = chunk.bm25_score
        elif isinstance(chunk, HybridChunk):
            score = chunk.rrf_score
        
        sources.append(
            Source(
                document=chunk.document_name,
                page=chunk.page_number,
                chunk_id=chunk.chunk_id,
                similarity_score=score,
            )
        )

    total_latency_ms = tracer.retrieval_ms + tracer.reranker_ms + tracer.generation_ms + tracer.evaluation_ms

    return QueryResponseWithTrace(
        answer=answer,
        sources=sources,
        latency_ms=total_latency_ms,
        trace_id=trace_id,
        evaluation=evaluation_data,
        grounding=grounding_data,
    )
