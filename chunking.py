
from langchain_text_splitters import RecursiveCharacterTextSplitter


def split_documents(documents, chunk_size=800, chunk_overlap=150):
    """Split loaded documents into chunks while preserving metadata."""
    if chunk_size <= 0:
        raise ValueError("chunk_size must be greater than zero.")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be non-negative and smaller than chunk_size.")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        add_start_index=True,
    )

    chunks = []

    for document in documents:
        text = document.get("text", "")
        metadata = document.get("metadata", {})

        if not text.strip():
            continue

        for chunk in splitter.create_documents(
            [text],
            metadatas=[metadata],
        ):
            chunks.append({
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            })

    return chunks
