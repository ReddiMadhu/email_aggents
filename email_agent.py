"""
Email Agent Module
==================
LangGraph-based agent for document search and email sending.

Uses Outlook for email via win32com.
"""

import os
from typing import TypedDict, List, Dict, Optional

from langgraph.graph import StateGraph, END
import dotenv

dotenv.load_dotenv()

# Import storage module
try:
    import cloud_storage as storage
except ImportError:
    import mock_storage as storage

# Import Outlook email module
from outlook_email import send_document_email, is_outlook_available

# =============================================================================
# Constants
# =============================================================================

VALID_ACCOUNTS = ["Amex", "Chubbs", "GlobalInsure", "TechCorp", "Travelers", "WESLACO_ISD"]
VALID_LOBS = ["AUTO", "PROPERTY", "GL", "WC", "InlandMarine"]

LOB_EMAILS = {
    "AUTO": os.getenv("AUTO_EMAIL", "claims-auto@company.com"),
    "PROPERTY": os.getenv("PROPERTY_EMAIL", "claims-property@company.com"),
    "GL": os.getenv("GL_EMAIL", "claims-gl@company.com"),
    "WC": os.getenv("WC_EMAIL", "claims-wc@company.com"),
    "InlandMarine": os.getenv("INLANDMARINE_EMAIL", "claims-marine@company.com"),
    "UNKNOWN": os.getenv("DEFAULT_EMAIL", "claims@company.com")
}

# =============================================================================
# State Definition
# =============================================================================

class AgentState(TypedDict):
    user_query: str
    account_name: Optional[str]
    insured_name: Optional[str]
    policy_number: Optional[str]
    lob: Optional[str]
    start_year: Optional[str]
    end_year: Optional[str]
    found_files: List[Dict[str, str]]
    validated_files: List[Dict[str, str]]
    email_sent: bool
    error: Optional[str]
    logs: List[str]

# =============================================================================
# Agent Nodes
# =============================================================================

def extract_info_node(state: AgentState) -> Dict:
    """Extract search criteria from user query using Gemini."""
    query = state['user_query']
    
    if not os.getenv("GOOGLE_API_KEY"):
        return {
            "error": "GOOGLE_API_KEY not found",
            "logs": state['logs'] + ["Error: Missing GOOGLE_API_KEY"]
        }

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from pydantic import BaseModel, Field
        
        class ExtractionResult(BaseModel):
            account_name: str = Field(description="Account name")
            insured_name: str = Field(description="Insured name if different from account")
            policy_number: str = Field(description="Policy number")
            lob: str = Field(description="Line of Business (AUTO, PROPERTY, GL, WC, InlandMarine)")
            start_year: str = Field(description="Start year (YYYY format)")
            end_year: str = Field(description="End year (YYYY format)")

        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        parser = JsonOutputParser(pydantic_object=ExtractionResult)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Extract document search criteria from user query.
Valid Accounts: {', '.join(VALID_ACCOUNTS)}
Valid LOBs: {', '.join(VALID_LOBS)}

Rules:
- Map variations: "chubbs/chubb" → "Chubbs", "amex" → "Amex", "weslaco" → "WESLACO_ISD"
- Map LOBs: "auto/vehicle" → "AUTO", "work/wc" → "WC", "property/home" → "PROPERTY", "gl/liability" → "GL", "marine/inland" → "InlandMarine"
- Extract year ranges: "2023-2024" → start_year: "2023", end_year: "2024"
- If only one year given, use it for both start and end
- "loss run" or "loss runs" indicates looking for loss run reports

{{format_instructions}}"""),
            ("user", "{query}")
        ])
        
        chain = prompt | llm | parser
        result = chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
        
        return {
            "account_name": result.get("account_name"),
            "insured_name": result.get("insured_name"),
            "policy_number": result.get("policy_number"),
            "lob": result.get("lob", "").upper() if result.get("lob") else None,
            "start_year": result.get("start_year"),
            "end_year": result.get("end_year"),
            "logs": state['logs'] + [f"Extracted: {result}"]
        }
    except Exception as e:
        return {
            "error": f"Extraction failed: {str(e)}",
            "logs": state['logs'] + [f"Extraction Error: {str(e)}"]
        }


def locate_document_node(state: AgentState) -> Dict:
    """
    Search for documents by folder structure and filenames:
    1. Search folder names (account, lob, policy)
    2. Search filenames (loss_run pattern with year range)
    """
    if state.get("error"):
        return state

    # Use the search function
    found_files = storage.search_files(
        account_name=state.get('account_name'),
        lob=state.get('lob'),
        policy_number=state.get('policy_number'),
        insured_name=state.get('insured_name'),
        start_year=state.get('start_year'),
        end_year=state.get('end_year')
    )
    
    return {
        "found_files": found_files,
        "logs": state['logs'] + [f"Found {len(found_files)} files via folder/filename/content search"]
    }


def validate_documents_node(state: AgentState) -> Dict:
    """
    Validate found documents by checking PDF content matches criteria.
    """
    if state.get("error"):
        return state
    
    found_files = state.get('found_files', [])
    validated_files = []
    
    for file_info in found_files:
        file_path = file_info.get('path', file_info.get('full_path', ''))
        
        # Use validation function if available
        if hasattr(storage, 'validate_document'):
            date_range = None
            if state.get('start_year') and state.get('end_year'):
                date_range = (state['start_year'], state['end_year'])
            
            validation = storage.validate_document(
                file_path,
                policy_number=state.get('policy_number'),
                date_range=date_range
            )
            
            if validation.get('valid', True):
                file_info['validated'] = True
                validated_files.append(file_info)
            else:
                # Still include but mark as unvalidated
                file_info['validated'] = False
                file_info['validation_errors'] = validation.get('errors', [])
                validated_files.append(file_info)
        else:
            # No validation available, include all
            file_info['validated'] = True
            validated_files.append(file_info)
    
    return {
        "validated_files": validated_files,
        "logs": state['logs'] + [f"Validated {len(validated_files)} files"]
    }

# =============================================================================
# Email Functions
# =============================================================================

def send_email_action(
    file_info: Dict[str, str], 
    lob: str, 
    policy_number: str, 
    recipient_override: str = None,
    cc_emails: str = None,
    excel_attachment: bytes = None,
    excel_filename: str = None,
    include_pdf: bool = True
) -> tuple:
    """
    Send email with document attachments via Outlook.
    
    Args:
        file_info: File information dictionary
        lob: Line of Business
        policy_number: Policy number
        recipient_override: Optional custom recipient email (overrides default LOB email)
        cc_emails: Optional CC email addresses (comma-separated)
        excel_attachment: Optional Excel file data as bytes
        excel_filename: Optional Excel filename
        include_pdf: Whether to include the original PDF
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    # Use override if provided, otherwise use default LOB email
    if recipient_override:
        recipient = recipient_override
    else:
        recipient = LOB_EMAILS.get(lob, LOB_EMAILS["UNKNOWN"])
    
    # Use Outlook email
    return send_document_email(
        file_info=file_info,
        lob=lob,
        policy_number=policy_number,
        recipient=recipient,
        cc_emails=cc_emails,
        excel_attachment=excel_attachment,
        excel_filename=excel_filename,
        include_pdf=include_pdf
    )

# =============================================================================
# Graph Construction
# =============================================================================

def build_graph():
    """Build the LangGraph workflow with validation step."""
    workflow = StateGraph(AgentState)
    
    workflow.add_node("extract_info", extract_info_node)
    workflow.add_node("locate_document", locate_document_node)
    workflow.add_node("validate_documents", validate_documents_node)
    
    workflow.set_entry_point("extract_info")
    workflow.add_edge("extract_info", "locate_document")
    workflow.add_edge("locate_document", "validate_documents")
    workflow.add_edge("validate_documents", END)
    
    return workflow.compile()


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    # Initialize mock data
    if hasattr(storage, 'create_mock_data'):
        storage.create_mock_data()

    # Test the agent
    app = build_graph()
    
    user_input = "chubbs auto policy 2456 loss run 2023-2024"
    print(f"\n🤖 User Query: '{user_input}'\n")
    
    initial_state = {
        "user_query": user_input,
        "account_name": None,
        "insured_name": None,
        "policy_number": None,
        "lob": None,
        "start_year": None,
        "end_year": None,
        "found_files": [],
        "validated_files": [],
        "email_sent": False,
        "error": None,
        "logs": []
    }
    
    result = app.invoke(initial_state)
    
    print("\n--- Result ---")
    if result.get("error"):
        print(f"❌ Error: {result['error']}")
    else:
        print(f"✅ Found {len(result['found_files'])} files")
        print(f"✅ Validated {len(result.get('validated_files', []))} files")
        for f in result.get('validated_files', result['found_files']):
            status = "✓" if f.get('validated', True) else "?"
            print(f"   [{status}] 📄 {f['filename']} ({f['account']}/{f['lob']}/{f['policy_number']})")
