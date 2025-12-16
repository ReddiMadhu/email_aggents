"""
EXL Claims Document Manager
===========================
Streamlit app for AI-powered insurance document processing.

Features:
1. Upload PDFs and extract claims data
2. Natural language document search
3. Document preview and email sending
4. Export results to Excel
"""

import os
import io
from datetime import datetime, date
from typing import Dict, List, Optional

import pandas as pd
import streamlit as st
import dotenv

dotenv.load_dotenv()

# Import utilities
from utils.pdf_utils import extract_text_from_pdf, preview_pdf
from utils.ai_parser import classify_lobs, extract_fields_chunked

# Import storage and email
try:
    import cloud_storage as storage
except ImportError:
    import mock_storage as storage

from email_agent import send_email_action, VALID_ACCOUNTS, VALID_LOBS, LOB_EMAILS

# =============================================================================
# API Setup - Use Google Gemini by default
# =============================================================================

def get_google_api_key() -> Optional[str]:
    """Get Google API key from environment."""
    return os.getenv("GOOGLE_API_KEY")

def setup_gemini_client():
    """Setup Google Gemini client."""
    api_key = get_google_api_key()
    if not api_key:
        return None
    try:
        import google.generativeai as genai
        genai.configure(api_key=api_key)
        return genai
    except Exception:
        return None

def get_model_name() -> str:
    """Get the default model name."""
    return "gemini-2.0-flash"

# =============================================================================
# Page Configuration
# =============================================================================

st.set_page_config(
    page_title="EXL Claims Document Manager",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# Custom CSS - Clean Modern Theme
# =============================================================================

st.markdown("""
<style>
    /* Modern Clean Theme with OKLCH colors */
    :root {
        --radius: 0.65rem;
        --background: oklch(1 0 0);
        --foreground: oklch(0.141 0.005 285.823);
        --card: oklch(1 0 0);
        --card-foreground: oklch(0.141 0.005 285.823);
        --primary: oklch(0.646 0.222 41.116);
        --primary-foreground: oklch(0.98 0.016 73.684);
        --secondary: oklch(0.967 0.001 286.375);
        --secondary-foreground: oklch(0.21 0.006 285.885);
        --muted: oklch(0.967 0.001 286.375);
        --muted-foreground: oklch(0.552 0.016 285.938);
        --accent: oklch(0.967 0.001 286.375);
        --accent-foreground: oklch(0.21 0.006 285.885);
        --border: oklch(0.92 0.004 286.32);
        --input: oklch(0.92 0.004 286.32);
        --ring: oklch(0.75 0.183 55.934);
        --sidebar: oklch(0.985 0 0);
        --sidebar-foreground: oklch(0.141 0.005 285.823);
        --sidebar-primary: oklch(0.646 0.222 41.116);
        --sidebar-border: oklch(0.92 0.004 286.32);
    }
    
    .stApp { 
        background-color: #d1d5db;
    }
    
    /* Hide Streamlit branding but keep sidebar toggle */
    #MainMenu, footer { visibility: hidden; }
    .stDeployButton { display: none; }
    
    /* Keep sidebar toggle button visible */
    [data-testid="collapsedControl"] {
        visibility: visible !important;
        display: flex !important;
    }
    
    /* Sidebar styling - Clean light */
    [data-testid="stSidebar"] {
        background-color: #e5e7eb;
        border-right: 1px solid #E5E5E5;
    }
    
    [data-testid="stSidebar"] * {
        color: #1A1A2E !important;
    }
    
    [data-testid="stSidebar"] .stMarkdown p {
        color: #666666 !important;
    }
    
    /* Button styling - Primary orange */
    .stButton > button {
        border-radius: var(--radius);
        font-weight: 500;
        transition: all 0.2s ease;
        border: 1px solid #E5E5E5;
    }
    
    .stButton > button[kind="primary"] {
        background-color: #E85D04;
        border: none;
        color: white !important;
    }
    
    .stButton > button[kind="primary"]:hover {
        background-color: #C54D00;
    }
    
    .stButton > button[kind="secondary"] {
        background-color: white;
        border: 1px solid #E5E5E5;
        color: #1A1A2E !important;
    }
    
    .stButton > button[kind="secondary"]:hover {
        background-color: #F5F5F5;
    }
    
    /* Card/Expander styling */
    .stExpander {
        background-color: white;
        border: 1px solid #E5E5E5;
        border-radius: var(--radius);
    }
    
    .stExpander:hover {
        border-color: #CCCCCC;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] { 
        color: #E85D04 !important; 
        font-weight: 600; 
    }
    
    /* Tabs */
    .stTabs [data-baseweb="tab-list"] {
        gap: 4px;
        background-color: #F5F5F5;
        padding: 4px;
        border-radius: var(--radius);
        width: 100%;
    }
    
    /* Make tabs fill full width */
    .stTabs [data-baseweb="tab-list"] button {
        flex: 1;
        width: 100%;
    }
    
    .stTabs [aria-selected="true"] { 
        background-color: white !important; 
        color: #1A1A2E !important;
        border-radius: calc(var(--radius) - 2px);
        box-shadow: 0 1px 3px rgba(0,0,0,0.1);
    }
    
    .stTabs [aria-selected="false"] {
        background-color: transparent;
        color: #666666;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        border: 1px dashed #CCCCCC;
        border-radius: var(--radius);
        padding: 1rem;
        background-color: #FAFAFA;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #E85D04;
        background-color: #FFF8F0;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background-color: #E85D04;
    }
    
    /* Input fields */
    .stTextInput > div > div > input,
    .stSelectbox > div > div > div {
        border-color: #E5E5E5;
        border-radius: var(--radius);
    }
    
    .stTextInput > div > div > input:focus,
    .stSelectbox > div > div > div:focus {
        border-color: #E85D04;
        box-shadow: 0 0 0 1px #E85D04;
    }
    
    /* Dataframe styling */
    .stDataFrame {
        border-radius: var(--radius);
        overflow: hidden;
        border: 1px solid #E5E5E5;
    }
    
    /* Links */
    a { color: #E85D04 !important; }
    a:hover { color: #C54D00 !important; }
    
    /* Divider */
    hr {
        border-color: #E5E5E5;
    }
    
    /* Full height tabs - flex layout */
    .main .block-container {
        display: flex;
        flex-direction: column;
        height: calc(100vh - 80px);
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 100%;
    }
    
    /* Target Streamlit tabs container */
    [data-testid="stTabs"] {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        height: 100%;
    }
    
    /* Tab list header */
    [data-testid="stTabs"] > div:first-child {
        flex-shrink: 0;
    }
    
    /* Tab panels container */
    [data-testid="stTabs"] > div:nth-child(2) {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        overflow-y: auto;
    }
    
    /* Individual tab panel */
    [data-testid="stTabs"] [data-baseweb="tab-panel"] {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
        padding: 1rem 0;
    }
    
    /* Vertical block inside tab panel */
    [data-testid="stTabs"] [data-baseweb="tab-panel"] > div {
        flex: 1;
        display: flex;
        flex-direction: column;
        min-height: 0;
    }
    
    /* Main section height */
    section[data-testid="stSidebar"] + section .main {
        height: 100vh;
        overflow: hidden;
    }
    
    /* Ensure stVerticalBlock fills space */
    [data-testid="stVerticalBlock"] {
        flex: 1;
        display: flex;
        flex-direction: column;
    }
</style>
""", unsafe_allow_html=True)

# =============================================================================
# Session State
# =============================================================================

def init_session_state():
    """Initialize session state variables."""
    defaults = {
        'per_lob': {},
        'processed': False,
        'extraction_success': False,
        'extracted_info': {},
        'agent_files': [],
        'selected_agent_files': [],
        'parse_results': [],
        'mock_data_created': False
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

init_session_state()

# Initialize mock data once
if not st.session_state['mock_data_created']:
    if hasattr(storage, 'create_mock_data'):
        storage.create_mock_data()
    st.session_state['mock_data_created'] = True

# =============================================================================
# Helper Functions
# =============================================================================

def parse_date_str(date_str: str) -> Optional[date]:
    """Parse date string to date object."""
    if not date_str:
        return None
    for fmt in ["%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d"]:
        try:
            return datetime.strptime(date_str, fmt).date()
        except ValueError:
            continue
    return None


def results_to_dataframes(all_results: List[Dict]) -> Dict[str, pd.DataFrame]:
    """Convert processing results to DataFrames by LoB."""
    data = {'AUTO': [], 'PROPERTY': [], 'GL': [], 'WC': []}
    
    for result in all_results:
        lob = result['lob']
        if lob == "GENERAL LIABILITY":
            lob = "GL"
        
        carrier = result.get('carrier', '')
        fields = result.get('fields', {})
        source_file = result.get('source_file', '')
        
        for claim in fields.get('claims', []):
            row = {
                'evaluation_date': fields.get('evaluation_date', ''),
                'carrier': claim.get('carrier', '') or carrier,
                'claim_number': claim.get('claim_number', ''),
                'loss_date': claim.get('loss_date', ''),
                'source_file': source_file
            }
            
            # Add LOB-specific fields
            if lob == 'AUTO' or lob == 'PROPERTY':
                row.update({
                    'paid_loss': claim.get('paid_loss', ''),
                    'reserve': claim.get('reserve', ''),
                    'alae': claim.get('alae', '')
                })
            elif lob == 'GL':
                row.update({
                    'bi_paid_loss': claim.get('bi_paid_loss', ''),
                    'pd_paid_loss': claim.get('pd_paid_loss', ''),
                    'bi_reserve': claim.get('bi_reserve', ''),
                    'pd_reserve': claim.get('pd_reserve', ''),
                    'alae': claim.get('alae', '')
                })
            elif lob == 'WC':
                row.update({
                    'Indemnity_paid_loss': claim.get('Indemnity_paid_loss', ''),
                    'Medical_paid_loss': claim.get('Medical_paid_loss', ''),
                    'Indemnity_reserve': claim.get('Indemnity_reserve', ''),
                    'Medical_reserve': claim.get('Medical_reserve', ''),
                    'ALAE': claim.get('ALAE', '')
                })
            
            if lob in data:
                data[lob].append(row)
    
    return {lob: pd.DataFrame(rows) for lob, rows in data.items() if rows}


def create_excel_download(per_lob: Dict[str, pd.DataFrame]) -> bytes:
    """Create Excel file with all LoB sheets."""
    output = io.BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for lob, df in per_lob.items():
            if not df.empty:
                df.to_excel(writer, sheet_name=f"{lob.lower()}_claims", index=False)
    output.seek(0)
    return output.getvalue()

# =============================================================================
# Sidebar - Clean Design
# =============================================================================

def render_sidebar():
    """Render the clean sidebar component."""
    
    with st.sidebar:
        # Logo section - using image
        logo_path = os.path.join(os.path.dirname(__file__), "assets", "exl_logo.png")
        if os.path.exists(logo_path):
            st.image(logo_path, use_container_width=True)
        else:
            # Fallback to text logo
            st.markdown("""
            <div style="text-align: center; padding: 1rem 0 1.5rem 0;">
                <span style="color: #E85D04; font-size: 2rem; font-weight: 700;">EXL</span>
            </div>
            """, unsafe_allow_html=True)
        
        # App title
        st.markdown("""
        <div style="text-align: center; margin-bottom: 1.5rem;">
            <p style="color: #1A1A2E; margin: 0; font-size: 1rem; font-weight: 600;">
                Claims Document Manager
            </p>
            <p style="color: #888888; font-size: 0.8rem; margin-top: 0.25rem;">
                AI-Powered Document Processing
            </p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        # About section
        st.markdown("**About**")
        st.markdown("""
        <p style="font-size: 0.85rem; color: #666;">
        Platform for insurance claims document processing. 
        Extract structured data from PDFs using AI technology.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        
        
        # Footer
        st.markdown("""
        <div style="text-align: center; padding-top: 0.5rem;">
            <small style="color: #999;">© 2025 EXL Service</small>
        </div>
        """, unsafe_allow_html=True)

# =============================================================================
# Tab 1: PDF Parser
# =============================================================================

def render_pdf_parser_tab():
    """Render the PDF Parser tab."""
    st.markdown("Upload insurance claims PDFs to extract structured data.")
    
    api_key = get_google_api_key()
    
    # File uploader
    uploaded_files = st.file_uploader(
        "Choose PDF files",
        type=['pdf'],
        accept_multiple_files=True
    )
    
    if uploaded_files:
        st.info(f"{len(uploaded_files)} file(s) uploaded")
    
    # Process button
    if st.button("Process PDFs", type="primary", disabled=not uploaded_files or not api_key):
        if not api_key:
            st.error("GOOGLE_API_KEY not set in .env")
            return
        
        try:
            import google.generativeai as genai
            genai.configure(api_key=api_key)
            model = genai.GenerativeModel(get_model_name())
        except Exception as e:
            st.error(f"Failed to setup Gemini: {e}")
            return
        
        all_results = []
        progress = st.progress(0)
        status = st.empty()
        
        for i, pdf_file in enumerate(uploaded_files):
            status.info(f"Processing: {pdf_file.name}")
            
            text = extract_text_from_pdf(pdf_file)
            if not text:
                st.warning(f"No text in {pdf_file.name}")
                continue
            
            status.info(f"Detecting LOBs in {pdf_file.name}")
            lobs = classify_lobs(model, text)
            
            for lob in lobs:
                status.info(f"Extracting {lob} claims from {pdf_file.name}")
                fields = extract_fields_chunked(model, text, lob)
                
                all_results.append({
                    'lob': lob,
                    'carrier': fields.get('carrier', ''),
                    'fields': fields,
                    'source_file': pdf_file.name
                })
            
            progress.progress((i + 1) / len(uploaded_files))
        
        if all_results:
            st.session_state['per_lob'] = results_to_dataframes(all_results)
            st.session_state['processed'] = True
            status.success("Processing complete!")
        else:
            status.warning("No claims data found")
    
    # Display results
    if st.session_state.get('processed') and st.session_state.get('per_lob'):
        per_lob = st.session_state['per_lob']
        
        st.markdown("---")
        st.header("Results")
        
        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("AUTO", len(per_lob.get('AUTO', [])))
        with col2:
            st.metric("PROPERTY", len(per_lob.get('PROPERTY', [])))
        with col3:
            st.metric("GL", len(per_lob.get('GL', [])))
        with col4:
            st.metric("WC", len(per_lob.get('WC', [])))
        
        # Data tabs
        tabs = st.tabs(["AUTO", "PROPERTY", "GL", "WC"])
        
        for i, lob in enumerate(["AUTO", "PROPERTY", "GL", "WC"]):
            with tabs[i]:
                if lob in per_lob and not per_lob[lob].empty:
                    st.dataframe(per_lob[lob], use_container_width=True)
                else:
                    st.info(f"No {lob} claims found")
        
        # Download button
        if any(lob in per_lob and not per_lob[lob].empty for lob in ['AUTO', 'PROPERTY', 'GL', 'WC']):
            st.download_button(
                "Download Excel",
                data=create_excel_download(per_lob),
                file_name="claims_extracted.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )

# =============================================================================
# Tab 2: AI Document Agent
# =============================================================================

def extract_search_criteria(query: str) -> Dict:
    """Extract search criteria from natural language query."""
    if not get_google_api_key():
        return {}
    
    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from langchain_core.output_parsers import JsonOutputParser
        from pydantic import BaseModel, Field
        
        class ExtractedInfo(BaseModel):
            account_name: str = Field(default="", description="Account name")
            insured_name: str = Field(default="", description="Insured name if different from account")
            lob: str = Field(default="", description="Line of Business")
            policy_number: str = Field(default="", description="Policy number")
            start_year: str = Field(default="", description="Start year YYYY")
            end_year: str = Field(default="", description="End year YYYY")
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0)
        parser = JsonOutputParser(pydantic_object=ExtractedInfo)
        
        prompt = ChatPromptTemplate.from_messages([
            ("system", f"""Extract search criteria from user query.
Valid Accounts: {', '.join(VALID_ACCOUNTS)}
Valid LOBs: {', '.join(VALID_LOBS)}

Rules:
- Map variations: "chubbs/chubb" → "Chubbs", "amex" → "Amex", "weslaco" → "WESLACO_ISD"
- Map LOBs: "auto/vehicle" → "AUTO", "work/wc" → "WC", "marine/inland" → "InlandMarine"
- Extract year ranges: "2023-2024" → start_year: "2023", end_year: "2024"
- Single year: use for both start and end
- "loss run" or "loss runs" indicates searching for loss run reports

{{format_instructions}}"""),
            ("user", "{query}")
        ])
        
        chain = prompt | llm | parser
        return chain.invoke({"query": query, "format_instructions": parser.get_format_instructions()})
    except Exception:
        return {}


def render_ai_agent_tab():
    """Render the AI Agent tab."""
    st.markdown("Search documents using natural language and send via email.")
    
    # Only show search and filters if no documents fetched yet
    show_search_filters = not st.session_state.get('agent_files')
    
    if show_search_filters:
        # Search input
        st.subheader("Search Documents")
        user_query = st.text_input(
            "Enter search query:",
            placeholder="e.g., chubbs auto policy 2456 loss run 2023-2024"
        )
        
        col1, col2 = st.columns([1, 5])
        
        with col1:
            if st.button("Extract", type="primary"):
                if user_query:
                    with st.spinner("Extracting..."):
                        extracted = extract_search_criteria(user_query)
                        if extracted:
                            st.session_state['extracted_info'] = extracted
                            st.session_state['extraction_success'] = True
                            st.rerun()
        
        with col2:
            if st.button("Clear"):
                st.session_state['extracted_info'] = {}
                st.session_state['agent_files'] = []
                st.session_state['selected_agent_files'] = []
                st.session_state['parse_results'] = []
                st.rerun()
        
        if st.session_state.get('extraction_success'):
            st.toast("Extracted! Filters auto-filled.", icon="✅")
            st.session_state['extraction_success'] = False
        
        st.markdown("---")
        
        # Editable filters - Updated for new structure
        st.subheader("Filters")
    
    extracted = st.session_state.get('extracted_info', {})
    
    if show_search_filters:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            options = ["-- All --"] + VALID_ACCOUNTS
            idx = 0
            if extracted.get('account_name') in VALID_ACCOUNTS:
                idx = options.index(extracted['account_name'])
            selected_account = st.selectbox("Account", options, index=idx)
        
        with col2:
            options = ["-- All --"] + VALID_LOBS
            idx = 0
            lob_val = extracted.get('lob', '').upper()
            # Handle InlandMarine case sensitivity
            if lob_val == "INLANDMARINE":
                lob_val = "InlandMarine"
            if lob_val in VALID_LOBS:
                idx = options.index(lob_val)
            selected_lob = st.selectbox("LOB", options, index=idx)
        
        with col3:
            policy_number = st.text_input("Policy", value=extracted.get('policy_number', ''))
        
        # Year range filters instead of date range
        col_y1, col_y2 = st.columns(2)
        with col_y1:
            start_year = st.text_input("Start Year", value=extracted.get('start_year', ''), placeholder="e.g., 2023")
        with col_y2:
            end_year = st.text_input("End Year", value=extracted.get('end_year', ''), placeholder="e.g., 2024")
        
        st.markdown("---")
        
        # Fetch button
        if st.button("Fetch Documents", type="primary"):
            acct = selected_account if selected_account != "-- All --" else None
            lob = selected_lob if selected_lob != "-- All --" else None
            policy = policy_number or None
            
            # Use the search with year range
            results = storage.search_files(
                account_name=acct,
                lob=lob,
                policy_number=policy,
                start_year=start_year if start_year else None,
                end_year=end_year if end_year else None
            )
            
            st.session_state['agent_files'] = results
            st.session_state['selected_agent_files'] = []
            
            if results:
                st.toast(f"Found {len(results)} document(s)", icon="✅")
                st.rerun()
            else:
                st.warning("No documents found")
    
    # Display found documents
    if st.session_state.get('agent_files'):
        # New Search button to go back
        if st.button("New Search", type="secondary"):
            st.session_state['agent_files'] = []
            st.session_state['selected_agent_files'] = []
            st.session_state['parse_results'] = []
            st.session_state['extracted_info'] = {}
            st.rerun()
        
        st.markdown("---")
        st.subheader("Documents")
        
        files = st.session_state['agent_files']
        selected = st.session_state.get('selected_agent_files', [])
        
        for idx, f in enumerate(files):
            col_chk, col_info = st.columns([0.5, 9.5])
            
            with col_chk:
                is_sel = st.checkbox("", key=f"sel_{idx}", value=f in selected)
                if is_sel and f not in selected:
                    selected.append(f)
                elif not is_sel and f in selected:
                    selected.remove(f)
            
            with col_info:
                # Updated display to show new structure info
                date_range = f.get('date_range', f"{f.get('start_year', '')}-{f.get('end_year', '')}")
                with st.expander(f"{f.get('filename')} | {f.get('account')} | {f.get('lob')} | {date_range}"):
                    st.markdown(f"""
| Field | Value |
|-------|-------|
| **Account** | {f.get('account', 'N/A')} |
| **Insured** | {f.get('insured_name', f.get('account', 'N/A'))} |
| **LOB** | {f.get('lob', 'N/A')} |
| **Policy** | {f.get('policy_number', 'N/A')} |
| **Date Range** | {date_range} |
| **Folder** | {f.get('folder_path', 'N/A')} |
""")
                    if st.button(f"Preview", key=f"prev_{idx}"):
                        path = f.get('path', f.get('full_path', ''))
                        if os.path.exists(path):
                            text = preview_pdf(path)
                            st.text_area("Preview", text, height=200)
                        else:
                            st.warning("File not found locally")
        
        st.session_state['selected_agent_files'] = selected
        
        # Actions section
        st.markdown("---")
        st.subheader("Actions")
        st.info(f"{len(selected)} document(s) selected")
        
        # Email configuration section - show when documents are selected
        if len(selected) > 0:
            st.markdown("##### Email Recipients")
            
            # Collect unique LOBs from selected files and their default emails
            lob_emails_for_selected = {}
            for f in selected:
                lob = f.get('lob', 'UNKNOWN')
                if lob not in lob_emails_for_selected:
                    default_email = LOB_EMAILS.get(lob, LOB_EMAILS.get('UNKNOWN', 'claims@company.com'))
                    lob_emails_for_selected[lob] = default_email
            
            # Create editable email inputs for each LOB
            edited_emails = {}
            for lob, default_email in lob_emails_for_selected.items():
                col_lob, col_email = st.columns([1, 3])
                with col_lob:
                    st.markdown(f"**{lob}:**")
                with col_email:
                    edited_emails[lob] = st.text_input(
                        f"Email for {lob}",
                        value=default_email,
                        key=f"email_{lob}",
                        label_visibility="collapsed",
                        placeholder="Enter email address(es), comma-separated"
                    )
            
            # Additional recipients
            additional_emails = st.text_input(
                "Additional Recipients (CC)",
                value="",
                placeholder="Enter additional email addresses, comma-separated",
                key="additional_emails"
            )
            
            st.markdown("---")
        
        col_email, col_parse, _ = st.columns([1, 1, 4])
        
        with col_email:
            if st.button("Send Email", disabled=len(selected) == 0, type="primary"):
                for f in selected:
                    lob = f.get('lob', 'UNKNOWN')
                    # Use the edited email if available
                    recipient = edited_emails.get(lob, LOB_EMAILS.get(lob, LOB_EMAILS.get('UNKNOWN')))
                    success, msg = send_email_action(f, lob, f.get('policy_number', ''), recipient_override=recipient)
                    if success:
                        st.toast(f"{f.get('filename')}: {msg}", icon="✅")
                    else:
                        st.error(f"{f.get('filename')}: {msg}")
        
        with col_parse:
            api_key = get_google_api_key()
            if st.button("Parse PDF", disabled=len(selected) == 0 or not api_key, type="primary"):
                try:
                    import google.generativeai as genai
                    genai.configure(api_key=api_key)
                    model = genai.GenerativeModel(get_model_name())
                except Exception as e:
                    st.error(f"Failed to setup Gemini: {e}")
                    return
                
                parse_results = []
                all_parsed_results = []  # For Excel generation
                
                progress_bar = st.progress(0)
                status_text = st.empty()
                
                for idx, f in enumerate(selected):
                    path = f.get('path', f.get('full_path', ''))
                    status_text.info(f"Parsing: {f.get('filename')}")
                    
                    try:
                        text = extract_text_from_pdf(path)
                        if not text.strip():
                            parse_results.append({"filename": f.get('filename'), "error": "No text"})
                            continue
                        
                        lobs = classify_lobs(model, text)
                        results = []
                        for lob in lobs:
                            fields = extract_fields_chunked(model, text, lob)
                            results.append({
                                "lob": lob,
                                "carrier": fields.get("carrier", ""),
                                "evaluation_date": fields.get("evaluation_date", ""),
                                "claims": fields.get("claims", [])
                            })
                            # Store for Excel generation
                            all_parsed_results.append({
                                'lob': lob,
                                'carrier': fields.get('carrier', ''),
                                'fields': fields,
                                'source_file': f.get('filename')
                            })
                        
                        parse_results.append({
                            "filename": f.get('filename'),
                            "file_info": f,  # Store file info for email
                            "success": True,
                            "detected_lobs": lobs,
                            "results": results
                        })
                    except Exception as e:
                        parse_results.append({"filename": f.get('filename'), "error": str(e)})
                    
                    progress_bar.progress((idx + 1) / len(selected))
                
                status_text.success("Parsing complete!")
                st.session_state['parse_results'] = parse_results
                st.session_state['parsed_dataframes'] = results_to_dataframes(all_parsed_results)
                st.session_state['parsed_files_for_email'] = [r.get('file_info') for r in parse_results if r.get('success') and r.get('file_info')]
        
        # Parse results section
        if st.session_state.get('parse_results'):
            st.markdown("---")
            st.subheader("Parse Results")
            
            # Show parsed data
            for res in st.session_state['parse_results']:
                if res.get('error'):
                    st.error(f"{res.get('filename')}: {res['error']}")
                else:
                    with st.expander(f"{res.get('filename')} - LOBs: {', '.join(res.get('detected_lobs', []))}", expanded=True):
                        for lob_res in res.get('results', []):
                            st.markdown(f"**{lob_res.get('lob')}** | Carrier: {lob_res.get('carrier')} | Eval Date: {lob_res.get('evaluation_date', 'N/A')}")
                            claims = lob_res.get('claims', [])
                            if claims:
                                st.dataframe(pd.DataFrame(claims), use_container_width=True)
                            else:
                                st.info("No claims found for this LOB")
            
            # Download Excel button
            if st.session_state.get('parsed_dataframes'):
                per_lob = st.session_state['parsed_dataframes']
                if any(lob in per_lob and not per_lob[lob].empty for lob in ['AUTO', 'PROPERTY', 'GL', 'WC']):
                    st.markdown("---")
                    st.subheader("Download Parsed Data")
                    st.download_button(
                        "Download Excel Report",
                        data=create_excel_download(per_lob),
                        file_name=f"parsed_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                        type="primary"
                    )
            
            # Email section for sending original PDFs and parsed Excel
            parsed_files = st.session_state.get('parsed_files_for_email', [])
            if parsed_files:
                st.markdown("---")
                st.subheader("Send Documents via Email")
                st.info(f"{len(parsed_files)} parsed document(s) ready to send (Original PDFs + Parsed Excel Report)")
                
                # Collect unique LOBs and their emails
                lob_emails_parsed = {}
                for f in parsed_files:
                    lob = f.get('lob', 'UNKNOWN')
                    if lob not in lob_emails_parsed:
                        default_email = LOB_EMAILS.get(lob, LOB_EMAILS.get('UNKNOWN', 'claims@company.com'))
                        lob_emails_parsed[lob] = default_email
                
                # Editable email inputs
                edited_emails_parsed = {}
                for lob, default_email in lob_emails_parsed.items():
                    col_lob, col_email = st.columns([1, 3])
                    with col_lob:
                        st.markdown(f"**{lob}:**")
                    with col_email:
                        edited_emails_parsed[lob] = st.text_input(
                            f"Email for {lob} (parsed)",
                            value=default_email,
                            key=f"parsed_email_{lob}",
                            label_visibility="collapsed",
                            placeholder="Enter email address(es), comma-separated"
                        )
                
                # CC Recipients
                cc_emails_parsed = st.text_input(
                    "CC Recipients",
                    value="",
                    placeholder="Enter CC email addresses, comma-separated",
                    key="parsed_cc_emails"
                )
                
                # Checkbox options
                col_opt1, col_opt2 = st.columns(2)
                with col_opt1:
                    include_original_pdfs = st.checkbox("Include Original PDFs", value=True, key="include_pdfs")
                with col_opt2:
                    include_excel_report = st.checkbox("Include Parsed Excel Report", value=True, key="include_excel")
                
                # Send button
                if st.button("Send Documents to LOB Emails", type="primary", key="send_parsed_emails"):
                    # Generate Excel data if needed
                    excel_data = None
                    excel_filename = None
                    if include_excel_report and st.session_state.get('parsed_dataframes'):
                        per_lob = st.session_state['parsed_dataframes']
                        if any(lob in per_lob and not per_lob[lob].empty for lob in ['AUTO', 'PROPERTY', 'GL', 'WC']):
                            excel_data = create_excel_download(per_lob)
                            excel_filename = f"parsed_claims_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
                    
                    for f in parsed_files:
                        lob = f.get('lob', 'UNKNOWN')
                        recipient = edited_emails_parsed.get(lob, LOB_EMAILS.get(lob, LOB_EMAILS.get('UNKNOWN')))
                        
                        # Get PDF path if including original
                        pdf_path = None
                        if include_original_pdfs:
                            pdf_path = f.get('path', f.get('full_path', ''))
                        
                        success, msg = send_email_action(
                            f, 
                            lob, 
                            f.get('policy_number', ''), 
                            recipient_override=recipient,
                            cc_emails=cc_emails_parsed if cc_emails_parsed else None,
                            excel_attachment=excel_data if include_excel_report else None,
                            excel_filename=excel_filename,
                            include_pdf=include_original_pdfs
                        )
                        if success:
                            st.toast(f"{f.get('filename')}: {msg}", icon="✅")
                        else:
                            st.error(f"{f.get('filename')}: {msg}")

# =============================================================================
# Main
# =============================================================================

def main():
    # Sidebar
    render_sidebar()
    
    # Tabs - AI Agent first, PDF Parser second
    tab1, tab2 = st.tabs(["AI Agent", "PDF Parser"])
    
    with tab1:
        render_ai_agent_tab()
    
    with tab2:
        render_pdf_parser_tab()


if __name__ == "__main__":
    main()
