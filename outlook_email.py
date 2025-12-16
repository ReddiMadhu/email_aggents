"""
Outlook Email Module
====================
Send emails via Microsoft Outlook using win32com.

Requires: pywin32 (pip install pywin32)
"""

import os
from typing import Optional, List

try:
    import win32com.client as win32
    OUTLOOK_AVAILABLE = True
except ImportError:
    OUTLOOK_AVAILABLE = False


def send_outlook_email(
    to: str,
    subject: str,
    body: str,
    cc: str = None,
    attachments: List[str] = None,
    html_body: bool = False
) -> tuple:
    """
    Send email via Outlook.
    
    Args:
        to: Recipient email address(es), comma-separated
        subject: Email subject
        body: Email body text
        cc: CC recipients, comma-separated (optional)
        attachments: List of file paths to attach (optional)
        html_body: If True, body is HTML formatted
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not OUTLOOK_AVAILABLE:
        return False, "pywin32 not installed. Run: pip install pywin32"
    
    try:
        outlook = win32.Dispatch('Outlook.Application')
        mail = outlook.CreateItem(0)  # 0 = olMailItem
        
        mail.To = to
        mail.Subject = subject
        
        if html_body:
            mail.HTMLBody = body
        else:
            mail.Body = body
        
        if cc:
            mail.CC = cc
        
        # Add attachments
        if attachments:
            for attachment_path in attachments:
                if os.path.exists(attachment_path):
                    mail.Attachments.Add(attachment_path)
        
        mail.Send()
        
        cc_info = f" (CC: {cc})" if cc else ""
        return True, f"Email sent to {to}{cc_info}"
    
    except Exception as e:
        return False, f"Outlook error: {str(e)}"


def send_document_email(
    file_info: dict,
    lob: str,
    policy_number: str,
    recipient: str,
    cc_emails: str = None,
    excel_attachment: bytes = None,
    excel_filename: str = None,
    include_pdf: bool = True
) -> tuple:
    """
    Send document email with PDF and/or Excel attachments.
    
    Args:
        file_info: File information dictionary with 'path' or 'full_path'
        lob: Line of Business
        policy_number: Policy number
        recipient: Recipient email address
        cc_emails: CC email addresses (comma-separated)
        excel_attachment: Excel file data as bytes
        excel_filename: Excel filename
        include_pdf: Whether to include the original PDF
    
    Returns:
        Tuple of (success: bool, message: str)
    """
    if not OUTLOOK_AVAILABLE:
        return False, "pywin32 not installed. Run: pip install pywin32"
    
    try:
        outlook = win32.Dispatch('Outlook.Application')
        mail = outlook.CreateItem(0)
        
        mail.To = recipient
        mail.Subject = f"Claims Document for Policy {policy_number} ({lob})"
        
        # Build email body
        body_parts = [f"Please find attached the claims document(s) for Policy {policy_number}."]
        attachments_list = []
        
        if include_pdf:
            attachments_list.append("Original PDF document")
        if excel_attachment:
            attachments_list.append("Parsed Excel report")
        
        if attachments_list:
            body_parts.append(f"\nAttachments: {', '.join(attachments_list)}")
        
        mail.Body = '\n'.join(body_parts)
        
        # Add CC
        if cc_emails:
            mail.CC = cc_emails
        
        # Attach PDF if requested
        if include_pdf:
            pdf_path = file_info.get('path', file_info.get('full_path', ''))
            if pdf_path and os.path.exists(pdf_path):
                mail.Attachments.Add(os.path.abspath(pdf_path))
        
        # Attach Excel if provided
        if excel_attachment and excel_filename:
            # Save Excel to temp file for attachment
            import tempfile
            temp_dir = tempfile.gettempdir()
            excel_path = os.path.join(temp_dir, excel_filename)
            with open(excel_path, 'wb') as f:
                f.write(excel_attachment)
            mail.Attachments.Add(excel_path)
        
        mail.Send()
        
        cc_info = f" (CC: {cc_emails})" if cc_emails else ""
        return True, f"Email sent to {recipient}{cc_info}"
    
    except Exception as e:
        return False, f"Outlook error: {str(e)}"


def is_outlook_available() -> bool:
    """Check if Outlook is available."""
    if not OUTLOOK_AVAILABLE:
        return False
    try:
        outlook = win32.Dispatch('Outlook.Application')
        return True
    except Exception:
        return False
