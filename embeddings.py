
from sentence_transformers import SentenceTransformer
from config import EMBEDDING_MODEL


_model = None


def get_embedding_model():
    """Load the embedding model once and reuse it."""
    global _model

    if _model is None:
        _model = SentenceTransformer(EMBEDDING_MODEL)

    return _model


def embed_texts(texts):
    """Convert a list of text strings into FAISS-ready vectors."""
    if not texts:
        return []

    model = get_embedding_model()

    vectors = model.encode(
        texts,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )

    return vectors.astype("float32")


def embed_query(question):
    """Convert one question into a vector using the same model."""
    vectors = embed_texts([question])
    return vectors[0]
