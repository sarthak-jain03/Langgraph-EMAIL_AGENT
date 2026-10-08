import io


def extract_text_from_files(files) -> str:
    if not files:
        return ""

    parts = []
    for file in files:
        try:
            file.seek(0)
            text = _extract_single(file)
            file.seek(0)
            if text.strip():
                parts.append(f"=== Document: {file.name} ===\n{text.strip()}")
        except Exception as exc:
            parts.append(f"=== Document: {file.name} ===\n[Extraction failed: {exc}]")

    return "\n\n".join(parts)


def _extract_single(file) -> str:
    name = file.name.lower()
    if name.endswith(".pdf"):
        return _extract_pdf(file)
    elif name.endswith(".docx"):
        return _extract_docx(file)
    elif name.endswith(".doc"):
        return "[.doc format is not supported — please convert to .docx or .pdf]"
    else:
        return _extract_text(file)


def _extract_pdf(file) -> str:
    try:
        from pypdf import PdfReader
    except ImportError:
        return "[pypdf is not installed — run: pip install pypdf]"
    raw = file.read()
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            pages.append(page_text)
    return "\n".join(pages)


def _extract_docx(file) -> str:
    try:
        from docx import Document
    except ImportError:
        return "[python-docx is not installed — run: pip install python-docx]"
    raw = file.read()
    doc = Document(io.BytesIO(raw))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    return "\n".join(paragraphs)


def _extract_text(file) -> str:
    raw = file.read()
    if isinstance(raw, str):
        return raw
    return raw.decode("utf-8", errors="replace")
