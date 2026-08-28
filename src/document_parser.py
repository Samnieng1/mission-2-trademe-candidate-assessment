"""Utilities to parse uploaded documents (TXT, DOCX, PDF) into text.

Imports for heavy dependencies are performed lazily so the module can be
imported in environments where optional parsers are not installed.
"""
from __future__ import annotations

import io
from typing import Tuple, Optional


def parse_txt_stream(b: bytes) -> str:
    try:
        return b.decode("utf-8")
    except Exception:
        try:
            return b.decode("latin-1")
        except Exception:
            raise ValueError("Could not decode TXT file as UTF-8 or Latin-1")


def parse_docx_stream(b: bytes) -> str:
    try:
        # Lazy import to avoid hard dependency at module import time
        from docx import Document
    except Exception as e:
        raise ImportError("python-docx not installed (pip install python-docx)") from e

    bio = io.BytesIO(b)
    doc = Document(bio)
    parts = []
    for para in doc.paragraphs:
        parts.append(para.text)
    return "\n".join(parts)


def parse_pdf_stream(b: bytes) -> Tuple[str, bool]:
    """Return (text, is_scanned).

    `is_scanned` is True if the PDF appears to be a scanned/image PDF (very
    little extractable text). This is a heuristic: if extracted text length is
    below a small threshold, we consider it scanned.
    """
    try:
        # Lazy import
        from pdfminer.high_level import extract_text
    except Exception as e:
        raise ImportError("pdfminer.six not installed (pip install pdfminer.six)") from e

    bio = io.BytesIO(b)
    try:
        text = extract_text(bio)
    except Exception:
        text = ""

    # Heuristic: if extracted text is very short, treat as scanned PDF
    if not text or len(text.strip()) < 120:
        return "", True
    return text, False


def parse_uploaded_file(f) -> Tuple[Optional[str], Optional[str]]:
    """Parse a Streamlit uploaded file-like object.

    Returns (text, error). If parsing succeeds, error is None. If parsing fails,
    text is None and error is a human-friendly message.
    """
    if f is None:
        return None, None

    name = getattr(f, "name", "").lower()
    # Size check done by caller; still safe to do here
    try:
        size = getattr(f, "size", None)
        if size and size > 5 * 1024 * 1024:
            return None, "File too large (max 5 MB)"
    except Exception:
        pass

    raw = None
    try:
        raw = f.getvalue() if hasattr(f, "getvalue") else f.read()
    except Exception:
        try:
            raw = f.read()
        except Exception:
            return None, "Could not read uploaded file"

    if name.endswith(".txt"):
        try:
            return parse_txt_stream(raw), None
        except Exception as e:
            return None, str(e)
    if name.endswith(".docx"):
        try:
            return parse_docx_stream(raw), None
        except ImportError as e:
            return None, str(e)
        except Exception:
            return None, "Could not parse DOCX file"
    if name.endswith(".pdf"):
        try:
            text, is_scanned = parse_pdf_stream(raw)
            if is_scanned:
                return None, "Scanned PDF or no extractable text — please paste the text instead."
            return text, None
        except ImportError as e:
            return None, str(e)
        except Exception:
            return None, "Could not parse PDF file"

    return None, "Unsupported file type"
