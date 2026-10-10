
def retrieve_chunks(vector_store, question, top_k=3):
    """Return the most relevant document chunks for a question."""
    if not question or not question.strip():
        raise ValueError("Please enter a question.")

    results = vector_store.search(question.strip(), top_k=top_k)

    return results
