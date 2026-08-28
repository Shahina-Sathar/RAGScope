"""Traces page - View and analyze query traces."""
import streamlit as st
from utils.api_client import RAGScopeAPI


def render_traces(api_client: RAGScopeAPI):
    """Render traces page."""
    st.markdown('<p class="main-header">🔍 Traces</p>', unsafe_allow_html=True)
    st.markdown("View and analyze query execution traces")
    st.markdown("---")
    
    # Check API connection
    if not api_client.health_check():
        st.error("❌ Backend API is not connected")
        return
    
    # Controls
    col1, col2 = st.columns([3, 1])
    
    with col1:
        limit = st.slider(
            "Number of traces to display",
            min_value=5,
            max_value=100,
            value=20,
            step=5
        )
    
    with col2:
        if st.button("🔄 Refresh"):
            st.rerun()
    
    # Load traces
    try:
        traces = api_client.list_traces(limit=limit)
        
        if not traces:
            st.info("No traces found. Execute some queries first!")
            return
        
        # Display traces
        for trace_summary in traces:
            trace = api_client.get_trace(trace_summary["id"])
            trace_id = trace.get("id", "unknown")
            query = trace.get("original_query", "No query")
            latency = trace.get("latency", {})
            total_ms = latency.get("total_ms", 0)
            
            with st.expander(f"🔍 {query[:80]}... ({total_ms}ms)"):
                # Basic info
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"**Trace ID:** `{trace_id}`")
                    st.markdown(f"**Mode:** {trace.get('retrieval_mode', 'N/A')}")
                
                with col2:
                    st.markdown(f"**Latency:** {total_ms} ms")
                    st.markdown(f"**Sources:** {len(trace.get('retrieval_results', {}).get('chunks', []))}")
                
                with col3:
                    st.markdown(f"**Created:** {trace.get('created_at', 'N/A')[:19]}")
                    if trace.get("rewritten_query"):
                        st.markdown("**Rewritten:** ✅")
                
                # Query details
                st.markdown("**Original Query:**")
                st.text(trace.get("original_query", "N/A"))
                
                if trace.get("rewritten_query"):
                    st.markdown("**Rewritten Query:**")
                    st.text(trace.get("rewritten_query"))
                
                # Answer
                generation = trace.get("generation_result", {})
                answer = generation.get("answer", "No answer")
                st.markdown("**Answer:**")
                st.markdown(answer)
                
                # Retrieved chunks
                retrieval_results = trace.get("retrieval_results", {})
                chunks = retrieval_results.get("chunks", [])
                
                if chunks:
                    st.markdown("**Retrieved Chunks:**")
                    for i, chunk in enumerate(chunks, 1):
                        with st.expander(f"Chunk {i}: {chunk.get('document_name', 'Unknown')} (Page {chunk.get('page_number', 'N/A')})"):
                            st.markdown(f"**Chunk ID:** `{chunk.get('chunk_id', 'N/A')}`")
                            
                            # Scores
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                if chunk.get("vector_score") is not None:
                                    st.markdown(f"**Vector Score:** {chunk.get('vector_score'):.4f}")
                            with col2:
                                if chunk.get("bm25_score") is not None:
                                    st.markdown(f"**BM25 Score:** {chunk.get('bm25_score'):.4f}")
                            with col3:
                                if chunk.get("reranker_score") is not None:
                                    st.markdown(f"**Reranker Score:** {chunk.get('reranker_score'):.4f}")
                            
                            # Text
                            st.markdown("**Text:**")
                            st.text(chunk.get("text", "N/A"))
                
                # Reranker results
                reranker_results = trace.get("reranker_results", {})
                reranked_chunks = reranker_results.get("chunks", [])
                
                if reranked_chunks:
                    st.markdown("**Reranked Chunks:**")
                    for i, chunk in enumerate(reranked_chunks[:5], 1):
                        st.markdown(f"{i}. {chunk.get('chunk_id', 'N/A')} (Score: {chunk.get('reranker_score', 0):.4f})")
                
                # Evaluation results
                evaluation = trace.get("evaluation_result", {})
                if evaluation and any(evaluation.values()):
                    st.markdown("**Evaluation Metrics:**")
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        if evaluation.get("faithfulness_score") is not None:
                            st.metric("Faithfulness", f"{evaluation.get('faithfulness_score'):.2f}")
                    
                    with col2:
                        if evaluation.get("answer_relevancy_score") is not None:
                            st.metric("Relevancy", f"{evaluation.get('answer_relevancy_score'):.2f}")
                    
                    with col3:
                        if evaluation.get("context_precision_score") is not None:
                            st.metric("Precision", f"{evaluation.get('context_precision_score'):.2f}")
                    
                    with col4:
                        if evaluation.get("context_recall_score") is not None:
                            st.metric("Recall", f"{evaluation.get('context_recall_score'):.2f}")
                
                # Grounding results
                grounding = trace.get("grounding_result", {})
                if grounding and grounding.get("grounding_score") is not None:
                    st.markdown("**Grounding Analysis:**")
                    is_grounded = grounding.get("is_grounded", False)
                    score = grounding.get("grounding_score", 0)
                    
                    if is_grounded:
                        st.success(f"✅ Grounded (Score: {score:.2f})")
                    else:
                        st.warning(f"⚠️ Not grounded (Score: {score:.2f})")
                    
                    supported = grounding.get("supported_claims", [])
                    hallucinated = grounding.get("hallucinated_claims", [])
                    
                    if supported or hallucinated:
                        col1, col2 = st.columns(2)
                        with col1:
                            if supported:
                                st.markdown("**Supported:**")
                                for claim in supported:
                                    st.markdown(f"- {claim}")
                        with col2:
                            if hallucinated:
                                st.markdown("**Hallucinated:**")
                                for claim in hallucinated:
                                    st.markdown(f"- ⚠️ {claim}")
                
                # Latency breakdown
                st.markdown("**Latency Breakdown:**")
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    st.metric("Retrieval", f"{latency.get('retrieval_ms', 0)} ms")
                with col2:
                    st.metric("Reranker", f"{latency.get('reranker_ms', 0)} ms")
                with col3:
                    st.metric("Generation", f"{latency.get('generation_ms', 0)} ms")
                with col4:
                    st.metric("Evaluation", f"{latency.get('evaluation_ms', 0)} ms")
                
                # Raw trace
                with st.expander("🔧 Raw Trace Data"):
                    st.json(trace)
    
    except Exception as e:
        st.error(f"❌ Error loading traces: {e}")
