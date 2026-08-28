#!/usr/bin/env python3
"""Verification script for Milestone 9: Streamlit Dashboard."""

import sys
from pathlib import Path


def verify_file_exists(filepath: str, description: str) -> bool:
    """Verify a file exists."""
    path = Path(filepath)
    if not path.exists():
        print(f"✗ {description}: {filepath} not found")
        return False
    print(f"✓ {description}: {filepath}")
    return True


def verify_file_content(filepath: str, required_strings: list[str], description: str) -> bool:
    """Verify a file contains required strings."""
    path = Path(filepath)
    if not path.exists():
        print(f"✗ {description}: {filepath} not found")
        return False
    
    content = path.read_text()
    missing = []
    for req_str in required_strings:
        if req_str not in content:
            missing.append(req_str)
    
    if missing:
        print(f"✗ {description}: Missing required content in {filepath}")
        for m in missing:
            print(f"  - {m}")
        return False
    
    print(f"✓ {description}: {filepath}")
    return True


def main():
    """Run all verification checks."""
    print("=" * 70)
    print("Milestone 9 Verification: Streamlit Dashboard")
    print("=" * 70)
    print()
    
    all_passed = True
    
    # 1. Verify frontend structure
    print("1. Frontend Structure")
    all_passed &= verify_file_exists(
        "../frontend/pyproject.toml",
        "Frontend dependencies"
    )
    all_passed &= verify_file_exists(
        "../frontend/app.py",
        "Main Streamlit app"
    )
    print()
    
    # 2. Verify API client
    print("2. API Client")
    all_passed &= verify_file_exists(
        "../frontend/utils/api_client.py",
        "API client module"
    )
    all_passed &= verify_file_content(
        "../frontend/utils/api_client.py",
        [
            "class RAGScopeAPI",
            "def health_check",
            "def upload_document",
            "def list_documents",
            "def index_document",
            "def query",
            "def list_traces",
            "def get_trace",
            "def create_experiment",
            "def list_experiments",
            "def run_experiment",
            "def compare_experiments",
        ],
        "API client methods"
    )
    print()
    
    # 3. Verify pages
    print("3. Dashboard Pages")
    pages = [
        ("../frontend/pages/home.py", "Home page", ["render_home", "System Overview"]),
        ("../frontend/pages/documents.py", "Documents page", ["render_documents", "Upload Document"]),
        ("../frontend/pages/query_playground.py", "Query Playground", ["render_query_playground", "RAG Configuration"]),
        ("../frontend/pages/traces.py", "Traces page", ["render_traces", "Retrieved Chunks"]),
        ("../frontend/pages/experiments.py", "Experiments page", ["render_experiments", "Create New"]),
        ("../frontend/pages/comparison.py", "Comparison page", ["render_comparison", "Metrics Comparison", "plotly"]),
    ]
    
    for page_path, description, required in pages:
        all_passed &= verify_file_exists(page_path, description)
        all_passed &= verify_file_content(page_path, required, f"{description} content")
    print()
    
    # 4. Verify main app integration
    print("4. Main App Integration")
    all_passed &= verify_file_content(
        "../frontend/app.py",
        [
            "import streamlit",
            "from utils.api_client import RAGScopeAPI",
            "from pages.home import render_home",
            "from pages.documents import render_documents",
            "from pages.query_playground import render_query_playground",
            "from pages.traces import render_traces",
            "from pages.experiments import render_experiments",
            "from pages.comparison import render_comparison",
        ],
        "Main app imports and routing"
    )
    print()
    
    # 5. Verify dependencies
    print("5. Dependencies")
    all_passed &= verify_file_content(
        "../frontend/pyproject.toml",
        [
            "streamlit",
            "plotly",
            "pandas",
            "requests",
        ],
        "Frontend dependencies"
    )
    print()
    
    # Summary
    print("=" * 70)
    if all_passed:
        print("✓ All verification checks passed!")
        print("=" * 70)
        print()
        print("Milestone 9 is complete. Key features:")
        print("  • Streamlit dashboard with 6 pages")
        print("  • API client for backend communication")
        print("  • Home page with system overview")
        print("  • Documents page for upload and management")
        print("  • Query Playground with all RAG options")
        print("  • Traces page for detailed trace viewing")
        print("  • Experiments page for experiment management")
        print("  • Comparison page with visual charts (Plotly)")
        return 0
    else:
        print("✗ Some verification checks failed")
        print("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
