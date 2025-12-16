"""
Mock Storage Module
===================
Simulates cloud storage for local development and testing.

NEW Folder Structure:
/insurance_documents
 └── {AccountName}
     └── {LOB}
         └── {POLICY_NUMBER}
             └── loss_runs
                 └── loss_run_{start_year}_{end_year}.pdf
"""

import os
import re
from typing import List, Dict, Optional
from datetime import datetime

# Base directory for mock storage
MOCK_STORAGE_ROOT = os.path.join(os.path.dirname(__file__), "insurance_documents")

# =============================================================================
# Document Registry - New Structure
# =============================================================================

DOCUMENT_REGISTRY = [
    # Chubbs Account
    {"account": "Chubbs", "lob": "AUTO", "policy_number": "2456", "start_year": "2023", "end_year": "2024", "insured_name": "Chubbs Insurance Corp"},
    {"account": "Chubbs", "lob": "WC", "policy_number": "2456", "start_year": "2023", "end_year": "2024", "insured_name": "Chubbs Insurance Corp"},
    {"account": "Chubbs", "lob": "PROPERTY", "policy_number": "3344", "start_year": "2023", "end_year": "2024", "insured_name": "Chubbs Insurance Corp"},
    {"account": "Chubbs", "lob": "GL", "policy_number": "5678", "start_year": "2023", "end_year": "2024", "insured_name": "Chubbs Insurance Corp"},
    
    # Amex Account
    {"account": "Amex", "lob": "AUTO", "policy_number": "7890", "start_year": "2023", "end_year": "2024", "insured_name": "American Express Financial"},
    {"account": "Amex", "lob": "PROPERTY", "policy_number": "7890", "start_year": "2022", "end_year": "2024", "insured_name": "American Express Financial"},
    {"account": "Amex", "lob": "GL", "policy_number": "5555", "start_year": "2023", "end_year": "2024", "insured_name": "American Express Financial"},
    {"account": "Amex", "lob": "WC", "policy_number": "1234", "start_year": "2022", "end_year": "2024", "insured_name": "American Express Financial"},
    
    # TechCorp Account
    {"account": "TechCorp", "lob": "WC", "policy_number": "1234", "start_year": "2023", "end_year": "2024", "insured_name": "TechCorp Industries LLC"},
    {"account": "TechCorp", "lob": "GL", "policy_number": "1234", "start_year": "2023", "end_year": "2024", "insured_name": "TechCorp Industries LLC"},
    {"account": "TechCorp", "lob": "AUTO", "policy_number": "4444", "start_year": "2023", "end_year": "2024", "insured_name": "TechCorp Industries LLC"},
    
    # Travelers Account
    {"account": "Travelers", "lob": "GL", "policy_number": "9999", "start_year": "2022", "end_year": "2024", "insured_name": "Travelers Insurance Group"},
    {"account": "Travelers", "lob": "PROPERTY", "policy_number": "8888", "start_year": "2023", "end_year": "2024", "insured_name": "Travelers Insurance Group"},
    {"account": "Travelers", "lob": "AUTO", "policy_number": "7777", "start_year": "2023", "end_year": "2024", "insured_name": "Travelers Insurance Group"},
    
    # GlobalInsure Account
    {"account": "GlobalInsure", "lob": "AUTO", "policy_number": "9999", "start_year": "2023", "end_year": "2024", "insured_name": "GlobalInsure International"},
    {"account": "GlobalInsure", "lob": "PROPERTY", "policy_number": "8888", "start_year": "2022", "end_year": "2024", "insured_name": "GlobalInsure International"},
    {"account": "GlobalInsure", "lob": "WC", "policy_number": "6666", "start_year": "2023", "end_year": "2024", "insured_name": "GlobalInsure International"},
    
    # WESLACO_ISD Account (example from spec)
    {"account": "WESLACO_ISD", "lob": "InlandMarine", "policy_number": "79580818", "start_year": "2011", "end_year": "2016", "insured_name": "Weslaco Independent School District"},
    {"account": "WESLACO_ISD", "lob": "PROPERTY", "policy_number": "79580819", "start_year": "2015", "end_year": "2020", "insured_name": "Weslaco Independent School District"},
]

# LOB Aliases for flexible matching
LOB_ALIASES = {
    "work": "wc", "workers": "wc", "workers comp": "wc", "worker": "wc", "workerscomp": "wc",
    "vehicle": "auto", "car": "auto", "accident": "auto", "automobile": "auto", "fleet": "auto",
    "house": "property", "home": "property", "fire": "property", "building": "property", "real estate": "property",
    "general": "gl", "liability": "gl", "general liability": "gl", "generalliability": "gl",
    "inland": "inlandmarine", "marine": "inlandmarine", "inland marine": "inlandmarine"
}

# Account Aliases for flexible matching
ACCOUNT_ALIASES = {
    "chubb": "Chubbs", "chubbs": "Chubbs",
    "amex": "Amex", "american express": "Amex",
    "techcorp": "TechCorp", "tech": "TechCorp",
    "travelers": "Travelers", "traveler": "Travelers",
    "globalinsure": "GlobalInsure", "global": "GlobalInsure", "global insure": "GlobalInsure",
    "weslaco": "WESLACO_ISD", "weslaco_isd": "WESLACO_ISD", "weslaco isd": "WESLACO_ISD"
}

# =============================================================================
# Helper Functions
# =============================================================================

def _normalize_string(s: str) -> str:
    """Normalize string for matching."""
    return str(s).lower().strip() if s else ""


def _resolve_account(account_str: str) -> Optional[str]:
    """Resolve account name from alias or partial match."""
    if not account_str:
        return None
    
    normalized = _normalize_string(account_str)
    
    # Check direct alias match
    if normalized in ACCOUNT_ALIASES:
        return ACCOUNT_ALIASES[normalized]
    
    # Check if it's already a valid account name
    for doc in DOCUMENT_REGISTRY:
        if _normalize_string(doc["account"]) == normalized:
            return doc["account"]
    
    # Fuzzy partial match
    for doc in DOCUMENT_REGISTRY:
        if normalized in _normalize_string(doc["account"]) or _normalize_string(doc["account"]) in normalized:
            return doc["account"]
    
    return account_str


def _resolve_lob(lob_str: str) -> Optional[str]:
    """Resolve LOB from alias or partial match."""
    if not lob_str:
        return None
    
    normalized = _normalize_string(lob_str)
    
    # Check alias first
    if normalized in LOB_ALIASES:
        return LOB_ALIASES[normalized].upper()
    
    # Direct match
    valid_lobs = ["AUTO", "PROPERTY", "GL", "WC", "INLANDMARINE"]
    if normalized.upper() in valid_lobs:
        return normalized.upper()
    
    return lob_str.upper()


def _fuzzy_match(search_term: str, target: str) -> bool:
    """Flexible partial matching."""
    if not search_term:
        return True
    
    search_norm = _normalize_string(search_term)
    target_norm = _normalize_string(target)
    
    if search_norm in target_norm or target_norm in search_norm:
        return True
    
    search_clean = re.sub(r'[^a-z0-9]', '', search_norm)
    target_clean = re.sub(r'[^a-z0-9]', '', target_norm)
    
    return search_clean in target_clean or target_clean in search_clean


def _get_folder_path(doc: Dict) -> str:
    """
    Generate folder path: AccountName/LOB/POLICY_NUMBER/loss_runs
    """
    return f"{doc['account']}/{doc['lob']}/{doc['policy_number']}/loss_runs"


def _get_filename(doc: Dict) -> str:
    """
    Generate filename: loss_run_{start_year}_{end_year}.pdf
    """
    return f"loss_run_{doc['start_year']}_{doc['end_year']}.pdf"


def _get_full_path(doc: Dict) -> str:
    """Get full file path including filename."""
    folder = _get_folder_path(doc)
    filename = _get_filename(doc)
    return os.path.join(MOCK_STORAGE_ROOT, folder, filename)


def _scan_filesystem_for_documents() -> List[Dict]:
    """
    Scan the filesystem to discover documents not in the registry.
    Enables finding documents by actual folder/file structure.
    """
    discovered = []
    
    if not os.path.exists(MOCK_STORAGE_ROOT):
        return discovered
    
    for account_name in os.listdir(MOCK_STORAGE_ROOT):
        account_path = os.path.join(MOCK_STORAGE_ROOT, account_name)
        if not os.path.isdir(account_path):
            continue
        
        for lob in os.listdir(account_path):
            lob_path = os.path.join(account_path, lob)
            if not os.path.isdir(lob_path):
                continue
            
            for policy_number in os.listdir(lob_path):
                policy_path = os.path.join(lob_path, policy_number)
                if not os.path.isdir(policy_path):
                    continue
                
                loss_runs_path = os.path.join(policy_path, "loss_runs")
                if not os.path.isdir(loss_runs_path):
                    continue
                
                for filename in os.listdir(loss_runs_path):
                    if filename.endswith('.pdf'):
                        match = re.match(r'loss_run_(\d{4})_(\d{4})\.pdf', filename)
                        start_year = match.group(1) if match else ""
                        end_year = match.group(2) if match else ""
                        
                        discovered.append({
                            "account": account_name,
                            "lob": lob,
                            "policy_number": policy_number,
                            "start_year": start_year,
                            "end_year": end_year,
                            "filename": filename,
                            "full_path": os.path.join(loss_runs_path, filename),
                            "insured_name": account_name
                        })
    
    return discovered


# =============================================================================
# Public API - Robust Search Logic
# =============================================================================

def search_files(
    account_name: Optional[str] = None,
    lob: Optional[str] = None,
    policy_number: Optional[str] = None,
    date: Optional[str] = None,
    insured_name: Optional[str] = None,
    start_year: Optional[str] = None,
    end_year: Optional[str] = None
) -> List[Dict[str, str]]:
    """
    Search for files by folder structure and filenames.
    
    Search Strategy:
    1. Search folder names (account, lob, policy)
    2. Search filenames (loss_run pattern with year range)
    
    Args:
        account_name: Account/Insured name (partial match supported)
        lob: Line of Business (aliases supported)
        policy_number: Policy number (partial match supported)
        date: Date in any format (extracts year for matching)
        insured_name: Insured name (used for account matching)
        start_year: Start year for date range
        end_year: End year for date range
    
    Returns:
        List of matching file info dictionaries.
    """
    results = []
    
    # Resolve aliases
    resolved_account = _resolve_account(account_name)
    resolved_lob = _resolve_lob(lob)
    search_policy = _normalize_string(policy_number) if policy_number else None
    
    # Extract year from date if provided
    search_year = None
    if date:
        year_match = re.search(r'(\d{4})', date)
        if year_match:
            search_year = year_match.group(1)
    
    if start_year:
        search_year = start_year
    
    # PHASE 1: Search by folder structure (registry + filesystem scan)
    all_documents = DOCUMENT_REGISTRY.copy()
    filesystem_docs = _scan_filesystem_for_documents()
    
    for fs_doc in filesystem_docs:
        is_in_registry = any(
            doc.get('account') == fs_doc['account'] and 
            doc.get('lob') == fs_doc['lob'] and 
            doc.get('policy_number') == fs_doc['policy_number'] and
            doc.get('start_year') == fs_doc['start_year']
            for doc in DOCUMENT_REGISTRY
        )
        if not is_in_registry:
            all_documents.append(fs_doc)
    
    matched_docs = []
    
    for doc in all_documents:
        # Check Account (folder name match)
        if resolved_account:
            if not _fuzzy_match(_normalize_string(resolved_account), _normalize_string(doc["account"])):
                continue
        
        # Check LOB (folder name match)
        if resolved_lob:
            doc_lob = _normalize_string(doc["lob"])
            if resolved_lob.lower() != doc_lob:
                continue
        
        # Check Policy Number (folder name match)
        if search_policy:
            if not _fuzzy_match(search_policy, doc["policy_number"]):
                continue
        
        # Check Date Range (filename match)
        if search_year:
            doc_start = doc.get("start_year", "")
            doc_end = doc.get("end_year", "")
            
            if doc_start and doc_end:
                try:
                    if not (int(doc_start) <= int(search_year) <= int(doc_end)):
                        continue
                except ValueError:
                    pass
        
        matched_docs.append(doc)
    
    # Build result objects
    for doc in matched_docs:
        filename = doc.get('filename') or _get_filename(doc)
        folder_path = _get_folder_path(doc)
        full_path = doc.get('full_path') or _get_full_path(doc)
        
        results.append({
            "filename": filename,
            "folder_path": folder_path,
            "full_path": full_path,
            "path": full_path,
            "source": "Mock Storage",
            "account": doc["account"],
            "lob": doc["lob"],
            "policy_number": doc["policy_number"],
            "start_year": doc.get("start_year", ""),
            "end_year": doc.get("end_year", ""),
            "date_range": f"{doc.get('start_year', '')}-{doc.get('end_year', '')}",
            "insured_name": doc.get("insured_name", doc["account"])
        })
    
    return results


def download_file(file_info: Dict[str, str]) -> str:
    """Return the local file path (for mock storage, just returns the path)."""
    return file_info.get("path", file_info.get("full_path", ""))


def validate_document(file_path: str, policy_number: str = None, date_range: tuple = None) -> Dict:
    """
    Validate document content matches expected criteria.
    
    Args:
        file_path: Path to the PDF file
        policy_number: Expected policy number
        date_range: Tuple of (start_year, end_year)
    
    Returns:
        Dict with validation results
    """
    validation = {
        "valid": True,
        "policy_match": None,
        "date_match": None,
        "errors": []
    }
    
    if not os.path.exists(file_path):
        validation["valid"] = False
        validation["errors"].append("File not found")
        return validation
    
    try:
        from utils.pdf_utils import extract_text_from_pdf
        text = extract_text_from_pdf(file_path)
        
        if policy_number:
            if policy_number.lower() in text.lower():
                validation["policy_match"] = True
            else:
                validation["policy_match"] = False
                validation["errors"].append(f"Policy {policy_number} not found in document")
        
        if date_range:
            start_year, end_year = date_range
            if str(start_year) in text or str(end_year) in text:
                validation["date_match"] = True
            else:
                validation["date_match"] = False
                validation["errors"].append(f"Date range {start_year}-{end_year} not found in document")
        
        if validation["errors"]:
            validation["valid"] = False
        
    except Exception as e:
        validation["valid"] = False
        validation["errors"].append(f"Validation error: {str(e)}")
    
    return validation


def create_mock_data():
    """Create mock PDF files in the new storage directory structure."""
    os.makedirs(MOCK_STORAGE_ROOT, exist_ok=True)
    
    try:
        import fpdf
        use_fpdf = True
    except ImportError:
        use_fpdf = False
    
    for doc in DOCUMENT_REGISTRY:
        # Create folder structure: AccountName/LOB/POLICY_NUMBER/loss_runs/
        folder_path = os.path.join(MOCK_STORAGE_ROOT, _get_folder_path(doc))
        filename = _get_filename(doc)
        file_path = os.path.join(folder_path, filename)
        
        os.makedirs(folder_path, exist_ok=True)
        
        if not os.path.exists(file_path):
            if use_fpdf:
                pdf = fpdf.FPDF()
                pdf.add_page()
                pdf.set_font("Arial", "B", 16)
                pdf.cell(200, 10, txt="LOSS RUN REPORT", ln=1, align="C")
                pdf.ln(10)
                pdf.set_font("Arial", size=12)
                pdf.cell(200, 10, txt=f"Insured: {doc.get('insured_name', doc['account'])}", ln=1)
                pdf.cell(200, 10, txt=f"Account: {doc['account']}", ln=1)
                pdf.cell(200, 10, txt=f"Line of Business: {doc['lob']}", ln=1)
                pdf.cell(200, 10, txt=f"Policy Number: {doc['policy_number']}", ln=1)
                pdf.cell(200, 10, txt=f"Report Period: {doc['start_year']} - {doc['end_year']}", ln=1)
                pdf.ln(10)
                pdf.cell(200, 10, txt="=" * 50, ln=1)
                pdf.cell(200, 10, txt="Sample Claims Data", ln=1)
                pdf.cell(200, 10, txt=f"Claim #001 - Loss Date: 01-15-{doc['start_year']}", ln=1)
                pdf.cell(200, 10, txt=f"Claim #002 - Loss Date: 06-20-{doc['end_year']}", ln=1)
                pdf.output(file_path)
            else:
                with open(file_path, 'wb') as f:
                    content = f"""%PDF-1.4
1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj
2 0 obj << /Type /Pages /Kids [3 0 R] /Count 1 >> endobj
3 0 obj << /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] >> endobj
trailer << /Size 4 /Root 1 0 R >>
%%EOF
LOSS RUN REPORT
Insured: {doc.get('insured_name', doc['account'])}
Account: {doc['account']} | LOB: {doc['lob']} | Policy: {doc['policy_number']}
Report Period: {doc['start_year']} - {doc['end_year']}
"""
                    f.write(content.encode())


def get_storage_stats() -> Dict:
    """Get statistics about the document storage."""
    accounts = set()
    lobs = set()
    total_docs = len(DOCUMENT_REGISTRY)
    
    for doc in DOCUMENT_REGISTRY:
        accounts.add(doc['account'])
        lobs.add(doc['lob'])
    
    return {
        "total_documents": total_docs,
        "accounts": list(accounts),
        "account_count": len(accounts),
        "lobs": list(lobs),
        "lob_count": len(lobs),
        "storage_root": MOCK_STORAGE_ROOT
    }


if __name__ == "__main__":
    create_mock_data()
    stats = get_storage_stats()
    print(f"✅ Mock storage ready with {stats['total_documents']} documents")
    print(f"   Accounts: {', '.join(stats['accounts'])}")
    print(f"   LOBs: {', '.join(stats['lobs'])}")
    print(f"   Location: {stats['storage_root']}")
    
    # Test searches
    print("\n🔍 Test Search: chubbs auto")
    for f in search_files(account_name="chubbs", lob="auto"):
        print(f"   📄 {f['folder_path']}/{f['filename']}")
    
    print("\n🔍 Test Search: policy 2456")
    for f in search_files(policy_number="2456"):
        print(f"   📄 {f['account']}/{f['lob']}/{f['filename']}")
    
    print("\n🔍 Test Search: WESLACO_ISD")
    for f in search_files(account_name="weslaco"):
        print(f"   📄 {f['folder_path']}/{f['filename']}")
