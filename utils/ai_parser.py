"""
AI Parser Module
================
AI-powered document parsing using Google Gemini.
"""

import json
import os
import time
from typing import Dict, List, Optional
from dotenv import load_dotenv

from .pdf_utils import extract_text_from_pdf, chunk_text

load_dotenv()


# LOB Schema Definitions
LOB_SCHEMAS = {
    'AUTO': {
        "evaluation_date": "string",
        "carrier": "string",
        "claims": [{
            "claim_number": "string",
            "loss_date": "string",
            "paid_loss": "string",
            "reserve": "string",
            "alae": "string"
        }]
    },
    'PROPERTY': {
        "evaluation_date": "string",
        "carrier": "string",
        "claims": [{
            "claim_number": "string",
            "loss_date": "string",
            "paid_loss": "string",
            "reserve": "string",
            "alae": "string"
        }]
    },
    'GL': {
        "evaluation_date": "string",
        "carrier": "string",
        "claims": [{
            "claim_number": "string",
            "loss_date": "string",
            "bi_paid_loss": "string",
            "pd_paid_loss": "string",
            "bi_reserve": "string",
            "pd_reserve": "string",
            "alae": "string"
        }]
    },
    'WC': {
        "evaluation_date": "string",
        "carrier": "string",
        "claims": [{
            "claim_number": "string",
            "loss_date": "string",
            "Indemnity_paid_loss": "string",
            "Medical_paid_loss": "string",
            "Indemnity_reserve": "string",
            "Medical_reserve": "string",
            "ALAE": "string"
        }]
    }
}

# LOB detection keywords
LOB_KEYWORDS = {
    'AUTO': [" AUTO ", " AUTOMOBILE", " VEHICLE", " VIN ", " COLLISION", " COMPREHENSIVE", " LICENSE PLATE"],
    'GL': [" GENERAL LIABILITY", " GL ", " PREMISES", " PRODUCTS LIABILITY", " CGL "],
    'WC': [" WORKERS' COMP", " WORKERS COMP", " WC ", " TTD", " TPD", " INDEMNITY"],
    'PROPERTY': [" PROPERTY ", " DWELLING", " BUILDING", " CONTENTS", " FIRE", " THEFT"]
}


def classify_lobs(model, text: str) -> List[str]:
    """
    Classify Lines of Business from text content.
    
    Args:
        model: Gemini GenerativeModel instance
        text: Document text
    
    Returns:
        List of detected LOBs.
    """
    prompt = f"""
You are an insurance domain expert. Determine ALL Lines of Business (LoBs) present in the content.
Choose any that apply from exactly these values: AUTO, GENERAL LIABILITY, WC, PROPERTY.
Return STRICT JSON ONLY with no commentary and no markdown. Use double quotes and valid JSON.
Schema: {{"lobs": ["AUTO"|"GENERAL LIABILITY"|"WC"|"PROPERTY", ...]}}
Content:\n{text[:10000]}
"""
    
    try:
        response = model.generate_content(prompt)
        content = response.text.strip()
        
        # Clean markdown if present
        if content.startswith("```"):
            content = content.split("```")[1]
            if content.startswith("json"):
                content = content[4:]
        content = content.strip()
        
        obj = json.loads(content)
        lobs = obj.get('lobs') or []
        
        # Normalize and validate
        valid_lobs = {"AUTO", "GENERAL LIABILITY", "GL", "WC", "PROPERTY"}
        result = []
        for v in lobs:
            s = str(v).strip().upper()
            if s == "GENERAL LIABILITY":
                s = "GL"
            if s in valid_lobs and s not in result:
                result.append(s)
        
        if result:
            return result
    except Exception:
        pass
    
    # Fallback: keyword detection
    return _detect_lobs_by_keywords(text)


def _detect_lobs_by_keywords(text: str) -> List[str]:
    """Detect LOBs using keyword matching."""
    text_upper = text.upper()
    found = []
    
    for lob, keywords in LOB_KEYWORDS.items():
        if any(kw in text_upper for kw in keywords):
            found.append(lob)
    
    return found or ["AUTO"]


def extract_fields(model, text: str, lob: str) -> Dict:
    """
    Extract structured fields from text for a specific LoB.
    
    Args:
        model: Gemini GenerativeModel instance
        text: Document text
        lob: Line of Business
    
    Returns:
        Dictionary with extracted fields.
    """
    lob = lob.upper()
    if lob == "GENERAL LIABILITY":
        lob = "GL"
    
    schema = LOB_SCHEMAS.get(lob, LOB_SCHEMAS['AUTO'])
    
    prompt = f"""
Extract structured fields from the content for LoB={lob}.
Return STRICT JSON ONLY matching this schema with no commentary and no markdown fences:
{schema}
Rules: ISO dates if possible; keep amounts/strings as-is; empty string if missing; preserve row order.
IMPORTANT: Extract the carrier/company name from the content. This is critical.

Content:\n{text}
"""
    
    max_attempts = 3
    delay = 1.0
    
    for attempt in range(max_attempts):
        try:
            response = model.generate_content(prompt)
            content = response.text.strip()
            
            # Clean markdown if present
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            content = content.strip()
            
            obj = json.loads(content)
            
            if isinstance(obj, dict) and 'claims' in obj and isinstance(obj['claims'], list):
                obj.setdefault('evaluation_date', '')
                obj.setdefault('carrier', '')
                return obj
        except Exception:
            if attempt < max_attempts - 1:
                time.sleep(delay)
                delay *= 2
            continue
    
    return {"evaluation_date": "", "carrier": "", "claims": []}


def extract_fields_chunked(
    model, 
    text: str, 
    lob: str,
    progress_callback: Optional[callable] = None
) -> Dict:
    """
    Extract fields with chunking for long documents.
    
    Args:
        model: Gemini GenerativeModel instance
        text: Document text
        lob: Line of Business
        progress_callback: Optional callback(current, total) for progress
    
    Returns:
        Merged dictionary with extracted fields.
    """
    chunks = chunk_text(text)
    if not chunks:
        chunks = [text]
    
    merged = {"evaluation_date": "", "carrier": "", "claims": []}
    
    for idx, chunk in enumerate(chunks):
        if progress_callback:
            progress_callback(idx + 1, len(chunks))
        
        result = extract_fields(model, chunk, lob)
        
        # Merge results
        if result.get('evaluation_date') and not merged['evaluation_date']:
            merged['evaluation_date'] = result['evaluation_date']
        if result.get('carrier') and not merged['carrier']:
            merged['carrier'] = result['carrier']
        if isinstance(result.get('claims'), list):
            merged['claims'].extend(result['claims'])
        
        time.sleep(0.3)
    
    return merged


def parse_pdf_document(file_path: str) -> Dict:
    """
    Complete PDF parsing: extract text, detect LOBs, extract fields.
    
    Args:
        file_path: Path to PDF file
    
    Returns:
        Dictionary with parsing results.
    """
    try:
        import google.generativeai as genai
        
        # Get API key from environment
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            return {"error": "GOOGLE_API_KEY not set in .env"}
        
        # Setup Gemini
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel("gemini-2.0-flash")
        
        # Extract text
        text_content = extract_text_from_pdf(file_path)
        if not text_content.strip():
            return {"error": "Could not extract text from PDF"}
        
        # Classify LOBs
        lobs = classify_lobs(model, text_content)
        
        # Extract fields for each LOB
        results = []
        for lob in lobs:
            fields = extract_fields_chunked(model, text_content, lob)
            results.append({
                "lob": lob,
                "carrier": fields.get("carrier", ""),
                "evaluation_date": fields.get("evaluation_date", ""),
                "claims": fields.get("claims", [])
            })
        
        return {
            "success": True,
            "source_file": file_path,
            "detected_lobs": lobs,
            "results": results,
            "text_preview": text_content[:500] + "..." if len(text_content) > 500 else text_content
        }
        
    except Exception as e:
        return {"error": f"Parse error: {str(e)}"}
