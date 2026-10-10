
from pathlib import Path
from pypdf import PdfReader


def load_document(file_path):
    """Read a PDF or TXT file and return its text and source metadata."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()

    if suffix == ".txt":
        text = path.read_text(encoding="utf-8-sig")

    elif suffix == ".pdf":
        reader = PdfReader(str(path))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)

    else:
        raise ValueError("Unsupported file type. Please use PDF or TXT.")

    if not text.strip():
        raise ValueError(
            f"No readable text found in {path.name}. "
            "The PDF may be scanned or the file may be empty."
        )

    return {
        "text": text,
        "metadata": {"source": path.name}
    }
