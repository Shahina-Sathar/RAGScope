"""API client for communicating with RAGScope backend."""
import os
import requests
from typing import Optional, Dict, Any, List
import json


class RAGScopeAPI:
    """Client for RAGScope backend API."""
    
    def __init__(self, base_url: str = None):
        self.base_url = (base_url or os.getenv("RAGSCOPE_API_URL", "http://localhost:8000")).rstrip("/")
    
    def _request(self, method: str, endpoint: str, **kwargs) -> Dict[str, Any]:
        """Make HTTP request to API."""
        url = f"{self.base_url}{endpoint}"
        try:
            response = requests.request(method, url, timeout=300, **kwargs)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            raise Exception(f"API request failed: {str(e)}")
    
    # Health check
    def health_check(self) -> bool:
        """Check if API is healthy."""
        try:
            response = self._request("GET", "/health")
            return response.get("status") == "healthy"
        except:
            return False
    
    # Documents
    def upload_document(self, file_path: str, filename: str | None = None) -> Dict[str, Any]:
        """Upload a document."""
        display_name = filename or os.path.basename(file_path)
        with open(file_path, "rb") as f:
            files = {"file": (display_name, f, "application/pdf")}
            return self._request("POST", "/documents/upload", files=files)
    
    def list_documents(self) -> List[Dict[str, Any]]:
        """List all documents."""
        return self._request("GET", "/documents")
    
    def index_document(self, document_id: str) -> Dict[str, Any]:
        """Index a document."""
        return self._request("POST", f"/documents/index/{document_id}")
    
    def delete_document(self, document_id: str) -> Dict[str, Any]:
        """Delete a document."""
        return self._request("DELETE", f"/documents/{document_id}")
    
    # Query
    def query(
        self,
        question: str,
        top_k: int = 5,
        retrieval_mode: str = "vector",
        reranker_enabled: bool = False,
        reranker_top_k: int = 5,
        candidate_k: int = 20,
        query_rewriting_enabled: bool = False,
        query_rewriting_strategy: str = "none",
        corrective_rag_enabled: bool = False,
        corrective_rag_max_retries: int = 2,
        corrective_rag_relevance_threshold: float = 0.7,
        evaluation_enabled: bool = False,
        grounding_check_enabled: bool = False,
        grounding_threshold: float = 0.7,
    ) -> Dict[str, Any]:
        """Execute a query."""
        payload = {
            "question": question,
            "top_k": top_k,
            "retrieval_mode": retrieval_mode,
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
            "grounding_threshold": grounding_threshold,
        }
        return self._request("POST", "/query", json=payload)
    
    # Traces
    def list_traces(self, limit: int = 20) -> List[Dict[str, Any]]:
        """List traces."""
        return self._request("GET", f"/traces?limit={limit}")
    
    def get_trace(self, trace_id: str) -> Dict[str, Any]:
        """Get trace details."""
        return self._request("GET", f"/traces/{trace_id}")
    
    # Experiments
    def create_experiment(
        self,
        name: str,
        description: str,
        config: Dict[str, Any],
        dataset_path: str,
    ) -> Dict[str, Any]:
        """Create a new experiment."""
        payload = {
            "name": name,
            "description": description,
            "config": config,
            "dataset_path": dataset_path,
        }
        return self._request("POST", "/experiments", json=payload)
    
    def list_experiments(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        """List experiments."""
        params = f"?status={status}" if status else ""
        return self._request("GET", f"/experiments{params}")
    
    def get_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Get experiment details."""
        return self._request("GET", f"/experiments/{experiment_id}")
    
    def run_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Run an experiment."""
        return self._request("POST", f"/experiments/{experiment_id}/run")
    
    def get_experiment_results(
        self, experiment_id: str, failure_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get experiment results."""
        params = f"?failure_type={failure_type}" if failure_type else ""
        return self._request("GET", f"/experiments/{experiment_id}/results{params}")
    
    def analyze_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Analyze experiment failures."""
        return self._request("POST", f"/experiments/{experiment_id}/analyze")
    
    def get_failure_summary(self, experiment_id: str) -> Dict[str, Any]:
        """Get failure summary."""
        return self._request("GET", f"/experiments/{experiment_id}/failure-summary")
    
    def compare_experiments(self, experiment_ids: List[str]) -> Dict[str, Any]:
        """Compare experiments."""
        return self._request("POST", "/experiments/compare", json=experiment_ids)
    
    def delete_experiment(self, experiment_id: str) -> Dict[str, Any]:
        """Delete an experiment."""
        return self._request("DELETE", f"/experiments/{experiment_id}")
