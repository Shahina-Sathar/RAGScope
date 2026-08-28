"""
Grounding and hallucination detection for RAG systems.

This module provides functionality to detect whether an answer
is properly grounded in the retrieved context and identify
potential hallucinations.
"""
from dataclasses import dataclass
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage

from app.config.settings import settings


@dataclass
class GroundingResult:
    """Result of grounding check."""
    is_grounded: bool
    grounding_score: float  # 0.0 to 1.0
    hallucinated_claims: list[str]
    supported_claims: list[str]
    explanation: Optional[str] = None


class GroundingChecker:
    """
    Checks if an answer is grounded in the provided context.
    
    Uses LLM to analyze the answer and identify claims that are
    supported by the context vs. claims that appear to be hallucinated.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        """
        Initialize grounding checker.
        
        Args:
            llm: Language model for grounding checks (defaults to gpt-4o-mini)
        """
        self.llm = llm or ChatOpenAI(
            model=settings.openai_llm_model,
            api_key=settings.openai_api_key,
            temperature=0,
        )
    
    def check_grounding(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> GroundingResult:
        """
        Check if the answer is grounded in the context.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
        
        Returns:
            GroundingResult with grounding analysis
        """
        # Combine contexts into a single string
        combined_context = "\n\n---\n\n".join(contexts)
        
        system_prompt = """You are a fact-checking assistant. Your task is to analyze whether an answer is grounded in the provided context.

For each claim in the answer, determine:
1. Is it directly supported by the context?
2. Is it contradicted by the context?
3. Is it not mentioned in the context (potential hallucination)?

Be strict: only mark a claim as "supported" if there is clear evidence in the context.
If a claim cannot be verified from the context, mark it as hallucinated.

Respond in the following JSON format:
{
    "is_grounded": true/false,
    "grounding_score": 0.0-1.0,
    "supported_claims": ["claim 1", "claim 2"],
    "hallucinated_claims": ["claim 3"],
    "explanation": "Brief explanation of the grounding analysis"
}"""
        
        user_prompt = f"""Question: {question}

Context:
{combined_context}

Answer to analyze:
{answer}

Analyze whether the answer is grounded in the context. Respond with JSON only."""
        
        try:
            messages = [
                SystemMessage(content=system_prompt),
                HumanMessage(content=user_prompt),
            ]
            
            response = self.llm.invoke(messages)
            content = response.content.strip()
            
            # Parse JSON response
            import json
            # Handle potential markdown code blocks
            if content.startswith("```"):
                lines = content.split("\n")
                content = "\n".join(lines[1:-1])
            
            result_dict = json.loads(content)
            
            return GroundingResult(
                is_grounded=result_dict.get("is_grounded", False),
                grounding_score=result_dict.get("grounding_score", 0.0),
                supported_claims=result_dict.get("supported_claims", []),
                hallucinated_claims=result_dict.get("hallucinated_claims", []),
                explanation=result_dict.get("explanation"),
            )
        except Exception as e:
            # Return a conservative result on error
            return GroundingResult(
                is_grounded=False,
                grounding_score=0.0,
                supported_claims=[],
                hallucinated_claims=[answer],
                explanation=f"Grounding check failed: {str(e)}",
            )
    
    def is_grounded(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        threshold: float = 0.7,
    ) -> bool:
        """
        Quick check if answer is grounded above a threshold.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
            threshold: Minimum grounding score to consider grounded
        
        Returns:
            True if grounding score >= threshold, False otherwise
        """
        result = self.check_grounding(question, answer, contexts)
        return result.grounding_score >= threshold
    
    def get_hallucination_report(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> str:
        """
        Generate a human-readable hallucination report.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
        
        Returns:
            Formatted string report of grounding analysis
        """
        result = self.check_grounding(question, answer, contexts)
        
        report_lines = [
            f"Grounding Score: {result.grounding_score:.2f}",
            f"Is Grounded: {'Yes' if result.is_grounded else 'No'}",
            "",
            "Supported Claims:",
        ]
        
        if result.supported_claims:
            for claim in result.supported_claims:
                report_lines.append(f"  ✓ {claim}")
        else:
            report_lines.append("  (none)")
        
        report_lines.append("")
        report_lines.append("Hallucinated Claims:")
        
        if result.hallucinated_claims:
            for claim in result.hallucinated_claims:
                report_lines.append(f"  ✗ {claim}")
        else:
            report_lines.append("  (none)")
        
        if result.explanation:
            report_lines.append("")
            report_lines.append(f"Explanation: {result.explanation}")
        
        return "\n".join(report_lines)


class SimpleGroundingChecker:
    """
    Simple keyword-based grounding checker.
    
    Uses string matching to check if key terms from the answer
    appear in the context. Less accurate than LLM-based checking
    but faster and doesn't require additional API calls.
    """
    
    def check_grounding(
        self,
        question: str,
        answer: str,
        contexts: list[str],
    ) -> GroundingResult:
        """
        Check grounding using keyword matching.
        
        Args:
            question: The user's question
            answer: The generated answer
            contexts: List of retrieved context strings
        
        Returns:
            GroundingResult with keyword-based analysis
        """
        combined_context = " ".join(contexts).lower()
        
        # Extract key phrases from answer (simple approach)
        words = answer.lower().split()
        # Filter out common words
        stop_words = {"the", "a", "an", "is", "are", "was", "were", "be", "been", 
                     "being", "have", "has", "had", "do", "does", "did", "will", 
                     "would", "could", "should", "may", "might", "can", "to", "of",
                     "in", "on", "at", "by", "for", "with", "about", "against",
                     "between", "into", "through", "during", "before", "after",
                     "above", "below", "from", "up", "down", "and", "but", "or",
                     "nor", "not", "so", "yet", "both", "either", "neither",
                     "each", "every", "all", "any", "few", "more", "most", "other",
                     "some", "such", "no", "only", "own", "same", "than", "too",
                     "very", "just", "because", "as", "until", "while", "that",
                     "this", "these", "those", "i", "me", "my", "we", "our", "you",
                     "your", "he", "him", "his", "she", "her", "it", "its", "they",
                     "them", "their", "what", "which", "who", "whom", "when", "where",
                     "why", "how"}
        
        key_phrases = [w for w in words if w not in stop_words and len(w) > 3]
        
        if not key_phrases:
            return GroundingResult(
                is_grounded=True,
                grounding_score=1.0,
                supported_claims=[answer],
                hallucinated_claims=[],
                explanation="No key phrases to check",
            )
        
        # Check how many key phrases appear in context
        found_phrases = []
        missing_phrases = []
        
        for phrase in key_phrases:
            if phrase in combined_context:
                found_phrases.append(phrase)
            else:
                missing_phrases.append(phrase)
        
        grounding_score = len(found_phrases) / len(key_phrases) if key_phrases else 1.0
        
        return GroundingResult(
            is_grounded=grounding_score >= 0.7,
            grounding_score=grounding_score,
            supported_claims=found_phrases,
            hallucinated_claims=missing_phrases,
            explanation=f"Found {len(found_phrases)}/{len(key_phrases)} key phrases in context",
        )
