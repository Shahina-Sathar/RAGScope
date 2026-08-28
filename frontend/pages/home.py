"""Home page - System overview and quick stats."""
import streamlit as st
from utils.api_client import RAGScopeAPI


def render_home(api_client: RAGScopeAPI):
    """Render home page."""
    st.markdown('<p class="main-header">🔍 RAGScope Dashboard</p>', unsafe_allow_html=True)
    st.markdown("### RAG Evaluation & Observability Platform")
    st.markdown("---")
    
    # Check API connection
    is_healthy = api_client.health_check()
    
    if not is_healthy:
        st.error("❌ Backend API is not connected. Please start the backend server.")
        st.info("""
        **To start the backend:**
        ```bash
        cd backend
        uv run uvicorn app.main:app --reload
        ```
        """)
        return
    
    # Quick stats
    st.subheader("📊 System Overview")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        try:
            documents = api_client.list_documents()
            st.metric("Documents", len(documents))
        except:
            st.metric("Documents", "N/A")
    
    with col2:
        try:
            traces = api_client.list_traces(limit=1000)
            st.metric("Total Queries", len(traces))
        except:
            st.metric("Total Queries", "N/A")
    
    with col3:
        try:
            experiments = api_client.list_experiments()
            st.metric("Experiments", len(experiments))
        except:
            st.metric("Experiments", "N/A")
    
    with col4:
        try:
            completed = [e for e in experiments if e.get("status") == "completed"]
            st.metric("Completed", len(completed))
        except:
            st.metric("Completed", "N/A")
    
    st.markdown("---")
    
    # Recent activity
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕐 Recent Queries")
        try:
            traces = api_client.list_traces(limit=5)
            if traces:
                for trace in traces:
                    with st.expander(f"{trace.get('original_query', 'N/A')[:50]}..."):
                        st.json({
                            "Query": trace.get("original_query"),
                            "Mode": trace.get("retrieval_mode"),
                            "Latency": f"{trace.get('latency', {}).get('total_ms', 0)} ms",
                            "Sources": len(trace.get("retrieval_results", {}).get("chunks", [])),
                        })
            else:
                st.info("No queries yet. Try the Query Playground!")
        except Exception as e:
            st.error(f"Error loading traces: {e}")
    
    with col2:
        st.subheader("🧪 Recent Experiments")
        try:
            experiments = api_client.list_experiments()
            if experiments:
                for exp in experiments[:5]:
                    status = exp.get("status", "unknown")
                    status_color = {
                        "completed": "green",
                        "running": "blue",
                        "failed": "red",
                        "created": "gray",
                    }.get(status, "gray")
                    
                    with st.expander(f"{exp.get('name', 'N/A')}"):
                        st.markdown(f"**Status:** :{status_color}[{status}]")
                        st.markdown(f"**Queries:** {exp.get('total_queries', 0)}")
                        if exp.get("avg_latency_ms"):
                            st.markdown(f"**Avg Latency:** {exp.get('avg_latency_ms'):.0f} ms")
            else:
                st.info("No experiments yet. Create one in the Experiments page!")
        except Exception as e:
            st.error(f"Error loading experiments: {e}")
    
    st.markdown("---")
    
    # Quick start guide
    st.subheader("🚀 Quick Start")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **1. Upload Documents**
        
        Go to the Documents page to upload and index your PDF documents.
        """)
    
    with col2:
        st.markdown("""
        **2. Test Queries**
        
        Use the Query Playground to test different RAG configurations.
        """)
    
    with col3:
        st.markdown("""
        **3. Run Experiments**
        
        Create and run experiments to evaluate RAG performance.
        """)
