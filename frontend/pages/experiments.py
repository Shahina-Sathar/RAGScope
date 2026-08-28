"""Experiments page - Create, run, and manage experiments."""
import streamlit as st
from utils.api_client import RAGScopeAPI


def render_experiments(api_client: RAGScopeAPI):
    """Render experiments page."""
    st.markdown('<p class="main-header">🧪 Experiments</p>', unsafe_allow_html=True)
    st.markdown("Create, run, and manage RAG experiments")
    st.markdown("---")
    
    # Check API connection
    if not api_client.health_check():
        st.error("❌ Backend API is not connected")
        return
    
    # Tabs for different views
    tab1, tab2, tab3 = st.tabs(["📋 All Experiments", "➕ Create New", "📊 Results"])
    
    with tab1:
        # List all experiments
        st.subheader("All Experiments")
        
        if st.button("🔄 Refresh"):
            st.rerun()
        
        try:
            experiments = api_client.list_experiments()
            
            if not experiments:
                st.info("No experiments yet. Create one in the 'Create New' tab!")
            else:
                for exp in experiments:
                    exp_id = exp.get("id", "unknown")
                    name = exp.get("name", "Unnamed")
                    status = exp.get("status", "unknown")
                    
                    # Status badge
                    status_color = {
                        "completed": "green",
                        "running": "blue",
                        "failed": "red",
                        "created": "gray",
                    }.get(status, "gray")
                    
                    with st.expander(f"{name} :{status_color}[{status}]"):
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            st.markdown(f"**ID:** `{exp_id}`")
                            st.markdown(f"**Total Queries:** {exp.get('total_queries', 0)}")
                            st.markdown(f"**Completed:** {exp.get('completed_queries', 0)}")
                        
                        with col2:
                            if exp.get("avg_latency_ms"):
                                st.markdown(f"**Avg Latency:** {exp.get('avg_latency_ms'):.0f} ms")
                            if exp.get("avg_faithfulness"):
                                st.markdown(f"**Avg Faithfulness:** {exp.get('avg_faithfulness'):.2f}")
                            if exp.get("avg_grounding_score"):
                                st.markdown(f"**Avg Grounding:** {exp.get('avg_grounding_score'):.2f}")
                        
                        with col3:
                            st.markdown(f"**Created:** {exp.get('created_at', 'N/A')[:19]}")
                            if exp.get("completed_at"):
                                st.markdown(f"**Completed:** {exp.get('completed_at')[:19]}")
                        
                        # Actions
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            if status == "created":
                                if st.button("▶️ Run", key=f"run_{exp_id}"):
                                    try:
                                        api_client.run_experiment(exp_id)
                                        st.success("Experiment started!")
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error starting experiment: {e}")
                        
                        with col2:
                            if status == "completed":
                                if st.button("🔍 Analyze", key=f"analyze_{exp_id}"):
                                    try:
                                        analysis = api_client.analyze_experiment(exp_id)
                                        st.session_state[f"analysis_{exp_id}"] = analysis
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Error analyzing experiment: {e}")
                        
                        with col3:
                            if st.button("🗑️ Delete", key=f"delete_{exp_id}"):
                                try:
                                    api_client.delete_experiment(exp_id)
                                    st.success("Experiment deleted!")
                                    st.rerun()
                                except Exception as e:
                                    st.error(f"Error deleting experiment: {e}")
                        
                        # Show analysis if available
                        if f"analysis_{exp_id}" in st.session_state:
                            analysis = st.session_state[f"analysis_{exp_id}"]
                            st.markdown("**Failure Analysis:**")
                            
                            failure_types = analysis.get("failure_types", {})
                            if failure_types:
                                for failure_type, count in failure_types.items():
                                    st.markdown(f"- {failure_type}: {count}")
                            else:
                                st.success("No failures detected!")
        
        except Exception as e:
            st.error(f"❌ Error loading experiments: {e}")
    
    with tab2:
        # Create new experiment
        st.subheader("Create New Experiment")
        
        with st.form("create_experiment_form"):
            name = st.text_input("Experiment Name", placeholder="e.g., Baseline Vector Search")
            description = st.text_area("Description", placeholder="Describe the experiment...")
            dataset_path = st.text_input(
                "Dataset Path",
                value="/mnt/d/Work/RAGScope/evaluation/questions_mini.json",
                help="Absolute path to evaluation dataset JSON file"
            )
            
            st.markdown("**Configuration**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                retrieval_mode = st.selectbox(
                    "Retrieval Mode",
                    ["vector", "bm25", "hybrid"],
                    key="create_retrieval_mode"
                )
                top_k = st.slider("Top K", 1, 20, 5, key="create_top_k")
                reranker_enabled = st.checkbox("Enable Reranking", key="create_reranker")
                
                if reranker_enabled:
                    reranker_top_k = st.slider("Reranker Top K", 1, 20, 5, key="create_reranker_top_k")
                    candidate_k = st.slider("Candidate K", 5, 50, 20, key="create_candidate_k")
                else:
                    reranker_top_k = 5
                    candidate_k = 20
            
            with col2:
                query_rewriting_enabled = st.checkbox("Enable Query Rewriting", key="create_rewriting")
                
                if query_rewriting_enabled:
                    query_rewriting_strategy = st.selectbox(
                        "Rewriting Strategy",
                        ["none", "llm", "hyde", "decompose"],
                        key="create_rewriting_strategy"
                    )
                else:
                    query_rewriting_strategy = "none"
                
                corrective_rag_enabled = st.checkbox("Enable Corrective RAG", key="create_corrective")
                
                if corrective_rag_enabled:
                    corrective_rag_max_retries = st.slider(
                        "Max Retries", 0, 5, 2, key="create_max_retries"
                    )
                    corrective_rag_relevance_threshold = st.slider(
                        "Relevance Threshold", 0.0, 1.0, 0.7, 0.05, key="create_threshold"
                    )
                else:
                    corrective_rag_max_retries = 2
                    corrective_rag_relevance_threshold = 0.7
                
                evaluation_enabled = st.checkbox("Enable Evaluation", value=True, key="create_evaluation")
                grounding_check_enabled = st.checkbox("Enable Grounding Check", value=True, key="create_grounding")
            
            submitted = st.form_submit_button("🚀 Create Experiment", type="primary")
            
            if submitted:
                if not name:
                    st.error("Please enter an experiment name")
                else:
                    try:
                        config = {
                            "retrieval_mode": retrieval_mode,
                            "top_k": top_k,
                            "reranker_enabled": reranker_enabled,
                            "reranker_top_k": reranker_top_k,
                            "candidate_k": candidate_k,
                            "query_rewriting_enabled": query_rewriting_enabled,
                            "query_rewriting_strategy": query_rewriting_strategy,
                            "corrective_rag_enabled": corrective_rag_enabled,
                            "corrective_rag_max_retries": corrective_rag_max_retries,
                            "corrective_rag_relevance_threshold": corrective_rag_relevance_threshold,
                            "evaluation_enabled": evaluation_enabled,
                            "grounding_check_enabled": grounding_check_enabled,
                        }
                        
                        result = api_client.create_experiment(
                            name=name,
                            description=description,
                            config=config,
                            dataset_path=dataset_path,
                        )
                        
                        st.success(f"✅ Experiment created! ID: {result.get('id')}")
                        st.rerun()
                    
                    except Exception as e:
                        st.error(f"❌ Error creating experiment: {e}")
    
    with tab3:
        # View experiment results
        st.subheader("Experiment Results")
        
        try:
            experiments = api_client.list_experiments()
            completed_experiments = [e for e in experiments if e.get("status") == "completed"]
            
            if not completed_experiments:
                st.info("No completed experiments yet.")
            else:
                selected_exp = st.selectbox(
                    "Select Experiment",
                    [f"{e.get('name')} ({e.get('id')[:8]}...)" for e in completed_experiments],
                    key="results_select"
                )
                
                if selected_exp:
                    exp_id = completed_experiments[[f"{e.get('name')} ({e.get('id')[:8]}...)" for e in completed_experiments].index(selected_exp)].get("id")
                    
                    results = api_client.get_experiment_results(exp_id)
                    
                    if results:
                        st.markdown(f"**Total Results:** {len(results)}")
                        
                        # Filter by failure type
                        failure_types = set()
                        for r in results:
                            if r.get("failure_type"):
                                failure_types.add(r.get("failure_type"))
                        
                        if failure_types:
                            filter_type = st.selectbox(
                                "Filter by Failure Type",
                                ["All"] + list(failure_types),
                                key="filter_type"
                            )
                            
                            if filter_type != "All":
                                results = [r for r in results if r.get("failure_type") == filter_type]
                        
                        # Display results
                        for result in results:
                            with st.expander(f"{result.get('question', 'N/A')[:60]}..."):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.markdown("**Expected Answer:**")
                                    st.text(result.get("expected_answer", "N/A"))
                                
                                with col2:
                                    st.markdown("**Generated Answer:**")
                                    st.text(result.get("generated_answer", "N/A"))
                                
                                # Metrics
                                col1, col2, col3, col4 = st.columns(4)
                                
                                with col1:
                                    if result.get("faithfulness_score") is not None:
                                        st.metric("Faithfulness", f"{result.get('faithfulness_score'):.2f}")
                                
                                with col2:
                                    if result.get("relevancy_score") is not None:
                                        st.metric("Relevancy", f"{result.get('relevancy_score'):.2f}")
                                
                                with col3:
                                    if result.get("grounding_score") is not None:
                                        st.metric("Grounding", f"{result.get('grounding_score'):.2f}")
                                
                                with col4:
                                    st.metric("Latency", f"{result.get('latency_ms', 0)} ms")
                                
                                if result.get("failure_type"):
                                    st.warning(f"**Failure:** {result.get('failure_type')}")
                    
                    else:
                        st.info("No results for this experiment.")
        
        except Exception as e:
            st.error(f"❌ Error loading results: {e}")
