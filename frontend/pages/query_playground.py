"""Query Playground - Interactive query interface with all RAG options."""
import streamlit as st
from utils.api_client import RAGScopeAPI
import time


def render_query_playground(api_client: RAGScopeAPI):
    """Render query playground page."""
    st.markdown('<p class="main-header">💬 Query Playground</p>', unsafe_allow_html=True)
    st.markdown("Test different RAG configurations interactively")
    st.markdown("---")
    
    # Check API connection
    if not api_client.health_check():
        st.error("❌ Backend API is not connected")
        return
    
    # Query input
    st.subheader("🔍 Ask a Question")
    
    question = st.text_area(
        "Question",
        placeholder="Enter your question here...",
        height=100,
        help="Enter a question about your documents"
    )
    
    # Configuration options
    st.subheader("⚙️ RAG Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Retrieval Settings**")
        
        retrieval_mode = st.selectbox(
            "Retrieval Mode",
            ["vector", "bm25", "hybrid"],
            help="Vector: semantic search, BM25: keyword search, Hybrid: both combined"
        )
        
        top_k = st.slider(
            "Top K",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of chunks to retrieve"
        )
        
        reranker_enabled = st.checkbox(
            "Enable Reranking",
            help="Use cross-encoder to rerank retrieved chunks"
        )
        
        if reranker_enabled:
            reranker_top_k = st.slider(
                "Reranker Top K",
                min_value=1,
                max_value=20,
                value=5,
                help="Number of chunks to keep after reranking"
            )
            candidate_k = st.slider(
                "Candidate K",
                min_value=5,
                max_value=50,
                value=20,
                help="Number of candidates to retrieve before reranking"
            )
        else:
            reranker_top_k = 5
            candidate_k = 20
    
    with col2:
        st.markdown("**Advanced Features**")
        
        query_rewriting_enabled = st.checkbox(
            "Enable Query Rewriting",
            help="Rewrite query to improve retrieval"
        )
        
        if query_rewriting_enabled:
            query_rewriting_strategy = st.selectbox(
                "Rewriting Strategy",
                ["none", "llm", "hyde", "decompose"],
                help="Strategy for query rewriting"
            )
        else:
            query_rewriting_strategy = "none"
        
        corrective_rag_enabled = st.checkbox(
            "Enable Corrective RAG",
            help="Automatically retry with rewritten query if retrieval is poor"
        )
        
        if corrective_rag_enabled:
            corrective_rag_max_retries = st.slider(
                "Max Retries",
                min_value=0,
                max_value=5,
                value=2,
                help="Maximum number of retry attempts"
            )
            corrective_rag_relevance_threshold = st.slider(
                "Relevance Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum relevance score to accept retrieval"
            )
        else:
            corrective_rag_max_retries = 2
            corrective_rag_relevance_threshold = 0.7
        
        evaluation_enabled = st.checkbox(
            "Enable Evaluation",
            help="Calculate RAGAS metrics (faithfulness, relevancy, etc.)"
        )
        
        grounding_check_enabled = st.checkbox(
            "Enable Grounding Check",
            help="Check if answer is grounded in context"
        )
        
        if grounding_check_enabled:
            grounding_threshold = st.slider(
                "Grounding Threshold",
                min_value=0.0,
                max_value=1.0,
                value=0.7,
                step=0.05,
                help="Minimum grounding score to consider answer grounded"
            )
        else:
            grounding_threshold = 0.7
    
    st.markdown("---")
    
    # Execute query
    if st.button("🚀 Execute Query", type="primary", disabled=not question):
        if not question:
            st.warning("Please enter a question")
            return
        
        with st.spinner("Executing query..."):
            start_time = time.time()
            
            try:
                result = api_client.query(
                    question=question,
                    top_k=top_k,
                    retrieval_mode=retrieval_mode,
                    reranker_enabled=reranker_enabled,
                    reranker_top_k=reranker_top_k,
                    candidate_k=candidate_k,
                    query_rewriting_enabled=query_rewriting_enabled,
                    query_rewriting_strategy=query_rewriting_strategy,
                    corrective_rag_enabled=corrective_rag_enabled,
                    corrective_rag_max_retries=corrective_rag_max_retries,
                    corrective_rag_relevance_threshold=corrective_rag_relevance_threshold,
                    evaluation_enabled=evaluation_enabled,
                    grounding_check_enabled=grounding_check_enabled,
                    grounding_threshold=grounding_threshold,
                )
                
                elapsed_time = time.time() - start_time
                
                # Display results
                st.success(f"✅ Query completed in {elapsed_time:.2f}s")
                
                # Answer
                st.subheader("💡 Answer")
                st.markdown(result.get("answer", "No answer generated"))
                
                # Sources
                st.subheader("📚 Sources")
                sources = result.get("sources", [])
                if sources:
                    for i, source in enumerate(sources, 1):
                        with st.expander(f"Source {i}: {source.get('document', 'Unknown')} (Page {source.get('page', 'N/A')})"):
                            st.markdown(f"**Chunk ID:** `{source.get('chunk_id', 'N/A')}`")
                            st.markdown(f"**Score:** {source.get('similarity_score', 0):.4f}")
                else:
                    st.info("No sources retrieved")
                
                # Metadata
                st.subheader("📊 Metadata")
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("Total Latency", f"{result.get('latency_ms', 0)} ms")
                
                with col2:
                    st.metric("Sources Retrieved", len(sources))
                
                with col3:
                    trace_id = result.get("trace_id", "N/A")
                    st.metric("Trace ID", trace_id[:8] + "..." if trace_id != "N/A" else "N/A")
                
                # Evaluation results
                if evaluation_enabled and result.get("evaluation"):
                    st.subheader("📈 Evaluation Metrics")
                    eval_data = result.get("evaluation", {})
                    
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        faithfulness = eval_data.get("faithfulness_score")
                        if faithfulness is not None:
                            st.metric("Faithfulness", f"{faithfulness:.2f}")
                        else:
                            st.metric("Faithfulness", "N/A")
                    
                    with col2:
                        relevancy = eval_data.get("relevancy_score")
                        if relevancy is not None:
                            st.metric("Relevancy", f"{relevancy:.2f}")
                        else:
                            st.metric("Relevancy", "N/A")
                    
                    with col3:
                        precision = eval_data.get("precision_score")
                        if precision is not None:
                            st.metric("Precision", f"{precision:.2f}")
                        else:
                            st.metric("Precision", "N/A")
                    
                    with col4:
                        recall = eval_data.get("recall_score")
                        if recall is not None:
                            st.metric("Recall", f"{recall:.2f}")
                        else:
                            st.metric("Recall", "N/A")
                
                # Grounding results
                if grounding_check_enabled and result.get("grounding"):
                    st.subheader("🔒 Grounding Analysis")
                    grounding_data = result.get("grounding", {})
                    
                    is_grounded = grounding_data.get("is_grounded", False)
                    grounding_score = grounding_data.get("grounding_score", 0)
                    
                    if is_grounded:
                        st.success(f"✅ Answer is grounded (Score: {grounding_score:.2f})")
                    else:
                        st.warning(f"⚠️ Answer may not be fully grounded (Score: {grounding_score:.2f})")
                    
                    # Show claims
                    supported = grounding_data.get("supported_claims", [])
                    hallucinated = grounding_data.get("hallucinated_claims", [])
                    
                    if supported or hallucinated:
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            st.markdown("**Supported Claims:**")
                            if supported:
                                for claim in supported:
                                    st.markdown(f"- {claim}")
                            else:
                                st.info("No claims analyzed")
                        
                        with col2:
                            st.markdown("**Hallucinated Claims:**")
                            if hallucinated:
                                for claim in hallucinated:
                                    st.markdown(f"- ⚠️ {claim}")
                            else:
                                st.success("No hallucinations detected")
                
                # Raw result
                with st.expander("🔧 Raw Result"):
                    st.json(result)
                
            except Exception as e:
                st.error(f"❌ Error executing query: {e}")
