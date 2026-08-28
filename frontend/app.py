"""RAGScope Dashboard - Main Application."""
import streamlit as st
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from utils.api_client import RAGScopeAPI

# Page configuration
st.set_page_config(
    page_title="RAGScope Dashboard",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        text-align: center;
    }
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        font-weight: 500;
    }
    .status-completed {
        background-color: #d4edda;
        color: #155724;
    }
    .status-running {
        background-color: #cce5ff;
        color: #004085;
    }
    .status-failed {
        background-color: #f8d7da;
        color: #721c24;
    }
</style>
""", unsafe_allow_html=True)

# Initialize API client
@st.cache_resource
def get_api_client():
    """Get or create API client."""
    return RAGScopeAPI()

# Sidebar
st.sidebar.title("🔍 RAGScope")
st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation",
    ["🏠 Home", "📄 Documents", "💬 Query Playground", "🔍 Traces", "🧪 Experiments", "📊 Comparison"],
    index=0,
)

# API status check
api_client = get_api_client()
is_healthy = api_client.health_check()

if is_healthy:
    st.sidebar.success("✅ API Connected")
else:
    st.sidebar.error("❌ API Offline")
    st.sidebar.info("Make sure the backend is running on http://localhost:8000")

st.sidebar.markdown("---")
st.sidebar.markdown("**RAGScope v0.1.0**")
st.sidebar.markdown("RAG Evaluation & Observability Platform")

# Route to appropriate page
if page == "🏠 Home":
    from pages.home import render_home
    render_home(api_client)
elif page == "📄 Documents":
    from pages.documents import render_documents
    render_documents(api_client)
elif page == "💬 Query Playground":
    from pages.query_playground import render_query_playground
    render_query_playground(api_client)
elif page == "🔍 Traces":
    from pages.traces import render_traces
    render_traces(api_client)
elif page == "🧪 Experiments":
    from pages.experiments import render_experiments
    render_experiments(api_client)
elif page == "📊 Comparison":
    from pages.comparison import render_comparison
    render_comparison(api_client)
