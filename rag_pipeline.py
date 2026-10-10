
from document_loader import load_document
from chunking import split_documents
from vector_store import VectorStore
from retrieval import retrieve_chunks
from llm import generate_answer


class RAGPipeline:
    def __init__(self):
        self.vector_store = VectorStore()
        self.indexed = False

    def index_documents(self, file_paths):
        """Load documents, split them into chunks, and index them."""
        if not file_paths:
            raise ValueError("Please provide at least one document.")

        documents = [load_document(path) for path in file_paths]
        chunks = split_documents(documents)

        if not chunks:
            raise ValueError("No readable text chunks were found.")

        self.vector_store.build(chunks)
        self.indexed = True

        return len(chunks)

    def ask(self, question):
        """Retrieve relevant passages and generate an answer."""
        if not self.indexed:
            raise ValueError("Please process documents before asking questions.")

        if not question or not question.strip():
            raise ValueError("Please enter a question.")

        sources = retrieve_chunks(self.vector_store, question)

        answer = generate_answer(question.strip(), sources)

        return {
            "answer": answer,
            "sources": sources,
        }
