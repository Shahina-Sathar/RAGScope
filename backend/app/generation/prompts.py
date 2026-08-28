SYSTEM_PROMPT = """You are a helpful assistant that answers questions based on the provided context.

Your task is to:
1. Answer the user's question using ONLY the information from the provided context
2. Cite your sources by referencing the chunk IDs in square brackets, e.g., [chunk_1], [chunk_2]
3. If the answer cannot be found in the context, say "I couldn't find sufficient information in the provided documents to answer this question."

Context:
{context}"""

USER_PROMPT = """Question: {question}

Answer:"""


def format_context(chunks: list[dict]) -> str:
    context_parts = []
    for i, chunk in enumerate(chunks, 1):
        context_parts.append(
            f"[chunk_{i}] (Document: {chunk['document_name']}, Page: {chunk['page_number']}, "
            f"Chunk ID: {chunk['chunk_id']})\n{chunk['text']}"
        )
    return "\n\n".join(context_parts)
