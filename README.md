import fitz  # PyMuPDF

# Optional enhanced extractor
try:
    import pymupdf4llm
    HAS_PYMUPDF4LLM = True
except Exception:
    HAS_PYMUPDF4LLM = False


def convert_pdf_to_text(pdf_bytes: bytes) -> str:
    """
    Convert PDF bytes to extracted text (in-memory only).

    Args:
        pdf_bytes: PDF file content as bytes

    Returns:
        Extracted text string

    Raises:
        RuntimeError if extraction fails
    """

    try:
        # Open PDF directly from memory
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        # 1️⃣ Best quality: pymupdf4llm
        if HAS_PYMUPDF4LLM:
            try:
                text = pymupdf4llm.to_markdown(doc)
                if isinstance(text, str) and text.strip():
                    doc.close()
                    return text
            except Exception:
                pass

        # 2️⃣ Fallback: PyMuPDF page-by-page
        text_content = ""
        for page in doc:
            text_content += page.get_text()

        doc.close()

        if not text_content.strip():
            raise RuntimeError("No text extracted from PDF")

        return text_content

    except Exception as e:
        raise RuntimeError(f"PDF to text conversion failed: {e}")
st.markdown("""
<style>
/* Preview button custom outline style */
div[data-testid="stButton"] button[key="prev_selected"] {
    border: 2px solid var(--primary-color);
    color: var(--primary-color);
    background-color: transparent;
}

/* Hover effect (optional but recommended) */
div[data-testid="stButton"] button[key="prev_selected"]:hover {
    background-color: rgba(0, 0, 0, 0.03);
    color: var(--primary-color);
    border-color: var(--primary-color);
}
</style>
""", unsafe_allow_html=True)
