"""
Retrieval evaluation metrics for RAG systems.

All metrics are implemented transparently for inspection and testing.
"""


def hit_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> int:
    """
    Check if at least one relevant document appears in the top-K retrieved documents.
    
    Args:
        retrieved_ids: List of document IDs in retrieval order (rank 1 to N)
        relevant_ids: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        1 if at least one relevant document is in top-K, 0 otherwise
        
    Example:
        >>> hit_at_k(["doc1", "doc2", "doc3"], ["doc3", "doc5"], k=2)
        0
        >>> hit_at_k(["doc1", "doc2", "doc3"], ["doc3", "doc5"], k=3)
        1
    """
    if k <= 0:
        return 0
    
    top_k_retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    for doc_id in top_k_retrieved:
        if doc_id in relevant_set:
            return 1
    
    return 0


def recall_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate what fraction of relevant documents appear in the top-K retrieved documents.
    
    Args:
        retrieved_ids: List of document IDs in retrieval order (rank 1 to N)
        relevant_ids: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        Fraction of relevant documents found in top-K (0.0 to 1.0)
        
    Example:
        >>> recall_at_k(["doc1", "doc2", "doc3"], ["doc2", "doc3", "doc5"], k=3)
        0.6666666666666666
    """
    if not relevant_ids or k <= 0:
        return 0.0
    
    top_k_retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    found = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    return found / len(relevant_set)


def precision_at_k(retrieved_ids: list[str], relevant_ids: list[str], k: int) -> float:
    """
    Calculate what fraction of the top-K retrieved documents are relevant.
    
    Args:
        retrieved_ids: List of document IDs in retrieval order (rank 1 to N)
        relevant_ids: List of relevant document IDs (ground truth)
        k: Number of top results to consider
        
    Returns:
        Fraction of top-K documents that are relevant (0.0 to 1.0)
        
    Example:
        >>> precision_at_k(["doc1", "doc2", "doc3"], ["doc2", "doc4"], k=3)
        0.3333333333333333
    """
    if k <= 0:
        return 0.0
    
    top_k_retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    found = sum(1 for doc_id in top_k_retrieved if doc_id in relevant_set)
    
    return found / k


def reciprocal_rank(retrieved_ids: list[str], relevant_ids: list[str]) -> float:
    """
    Calculate the reciprocal rank of the first relevant document.
    
    Args:
        retrieved_ids: List of document IDs in retrieval order (rank 1 to N)
        relevant_ids: List of relevant document IDs (ground truth)
        
    Returns:
        1/rank of first relevant document, or 0.0 if none found
        
    Example:
        >>> reciprocal_rank(["doc1", "doc2", "doc3"], ["doc3"])
        0.3333333333333333
        >>> reciprocal_rank(["doc1", "doc2", "doc3"], ["doc5"])
        0.0
    """
    relevant_set = set(relevant_ids)
    
    for rank, doc_id in enumerate(retrieved_ids, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    
    return 0.0


def mean_reciprocal_rank(
    all_retrieved: list[list[str]],
    all_relevant: list[list[str]]
) -> float:
    """
    Calculate the Mean Reciprocal Rank (MRR) across multiple queries.
    
    Args:
        all_retrieved: List of retrieved document ID lists (one per query)
        all_relevant: List of relevant document ID lists (one per query)
        
    Returns:
        Average reciprocal rank across all queries (0.0 to 1.0)
        
    Example:
        >>> all_retrieved = [["doc1", "doc2"], ["doc3", "doc4"]]
        >>> all_relevant = [["doc2"], ["doc5"]]
        >>> mean_reciprocal_rank(all_retrieved, all_relevant)
        0.25
    """
    if not all_retrieved or len(all_retrieved) != len(all_relevant):
        return 0.0
    
    rr_sum = sum(
        reciprocal_rank(retrieved, relevant)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    )
    
    return rr_sum / len(all_retrieved)


def average_precision_at_k(
    retrieved_ids: list[str],
    relevant_ids: list[str],
    k: int
) -> float:
    """
    Calculate Average Precision at K for a single query.
    
    Args:
        retrieved_ids: List of document IDs in retrieval order
        relevant_ids: List of relevant document IDs
        k: Number of top results to consider
        
    Returns:
        Average precision at K (0.0 to 1.0)
    """
    if not relevant_ids or k <= 0:
        return 0.0
    
    top_k_retrieved = retrieved_ids[:k]
    relevant_set = set(relevant_ids)
    
    precision_sum = 0.0
    relevant_count = 0
    
    for i, doc_id in enumerate(top_k_retrieved, start=1):
        if doc_id in relevant_set:
            relevant_count += 1
            precision_at_i = relevant_count / i
            precision_sum += precision_at_i
    
    if relevant_count == 0:
        return 0.0
    
    return precision_sum / min(len(relevant_set), k)


def mean_average_precision_at_k(
    all_retrieved: list[list[str]],
    all_relevant: list[list[str]],
    k: int
) -> float:
    """
    Calculate Mean Average Precision at K across multiple queries.
    
    Args:
        all_retrieved: List of retrieved document ID lists
        all_relevant: List of relevant document ID lists
        k: Number of top results to consider
        
    Returns:
        MAP@K across all queries (0.0 to 1.0)
    """
    if not all_retrieved or len(all_retrieved) != len(all_relevant):
        return 0.0
    
    ap_sum = sum(
        average_precision_at_k(retrieved, relevant, k)
        for retrieved, relevant in zip(all_retrieved, all_relevant)
    )
    
    return ap_sum / len(all_retrieved)
