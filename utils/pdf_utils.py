"""
PDF Utilities
=============
Functions for PDF text extraction and processing.
"""

from typing import List
import pdfplumber


def extract_text_from_pdf(pdf_file) -> str:
    """
    Extract text content from PDF file.
    
    Args:
        pdf_file: File path or file-like object
    
    Returns:
        Extracted text content as string.
    """
    text_content = []
    
    try:
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                # Extract regular text
                page_text = page.extract_text()
                if page_text:
                    text_content.append(page_text)
                
                # Also extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table:
                        for row in table:
                            if row:
                                row_text = " | ".join([str(cell) if cell else "" for cell in row])
                                text_content.append(row_text)
    except Exception:
        return ""
    
    return "\n".join(text_content)


def chunk_text(text: str, max_chars: int = 15000, overlap_chars: int = 800) -> List[str]:
    """
    Split text into overlapping chunks for processing.
    
    Args:
        text: Input text to chunk
        max_chars: Maximum characters per chunk
        overlap_chars: Overlap between chunks
    
    Returns:
        List of text chunks.
    """
    chunks: List[str] = []
    
    if not text:
        return chunks
    
    start = 0
    n = len(text)
    
    while start < n:
        end = min(start + max_chars, n)
        
        # Try to break at newline
        if end < n:
            nl = text.rfind("\n", start, end)
            if nl != -1 and nl > start + 1000:
                end = nl
        
        chunks.append(text[start:end])
        
        if end >= n:
            break
        
        start = max(0, end - overlap_chars)
    
    return chunks


def preview_pdf(file_path: str, max_pages: int = 2, max_chars: int = 3000) -> str:
    """
    Get a preview of PDF content.
    
    Args:
        file_path: Path to PDF file
        max_pages: Maximum pages to preview
        max_chars: Maximum characters to return
    
    Returns:
        Preview text.
    """
    try:
        text = ""
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages[:max_pages]:
                text += (page.extract_text() or "") + "\n"
        return text[:max_chars]
    except Exception:
        return ""
