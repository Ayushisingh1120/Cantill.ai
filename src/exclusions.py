import re
import pandas as pd
from typing import Tuple, Optional

def extract_po_number(val: str) -> Optional[str]:
    """Extract standard 10-digit SAP PO number from various formats."""
    if pd.isna(val) or not val:
        return None
    s = str(val).strip()
    m = re.search(r'(\d{10})', s)
    if m:
        return m.group(1)
    return None

def check_exclusion_company_a_tool(
    row: pd.Series,
    erp_po_set: set,
    seen_tool_signatures: set,
    status_rules: dict
) -> Tuple[int, str]:
    """
    Exclusion logic for Company A Tool AP lines:
    1. Status checks (Cancelled -> CANCELLED, Not approved -> NOT_APPROVED)
    2. Internal tool duplicate check -> DUPLICATE
    3. ERP cross-system match -> DUPLICATE
    """
    status = str(row.get('status', '')).strip()
    
    # Check Cancelled
    if status in status_rules.get('cancelled', ['Cancelled']):
        return 0, 'CANCELLED'
    
    # Check Not Approved
    if status in status_rules.get('not_approved', ['On Hold', 'Rejected']):
        return 0, 'NOT_APPROVED'
    
    # Internal duplicate check (tool invoice exported twice)
    # Signature: invoice_number, line_text, gross_amount
    inv_num = str(row.get('invoice_number', '')).strip().lower()
    line_txt = str(row.get('line_text', '')).strip().lower()
    gross_amt = round(float(row.get('clean_gross', 0.0)), 2)
    sig = (inv_num, line_txt, gross_amt)
    
    if inv_num and sig in seen_tool_signatures:
        return 0, 'DUPLICATE'
    if inv_num:
        seen_tool_signatures.add(sig)

    # Check duplicate against ERP primary records
    po_ref = row.get('po_reference')
    clean_po = extract_po_number(po_ref)
    if clean_po and clean_po in erp_po_set:
        return 0, 'DUPLICATE'
    
    # Valid included spend
    return 1, ''

def check_exclusion_company_b(
    row: pd.Series,
    exclusion_config: dict
) -> Tuple[int, str]:
    """
    Exclusion logic for Company B:
    - Non-procurement spend:
      - Intercompany GL descriptions (starting with IC or I/C)
      - Tax lines (sales tax, GST input tax, state & local taxes)
    """
    # 1. Check GL account description for intercompany
    gl_desc = str(row.get('gl_account_description', '')).strip().upper()
    ic_patterns = exclusion_config.get('intercompany_gl_patterns', [r'^IC', r'^I/C'])
    for pat in ic_patterns:
        if re.search(pat, gl_desc):
            return 0, 'NON_PROCUREMENT'
    
    # 2. Check line description for taxes
    line_desc = str(row.get('line_description', '')).strip().lower()
    tax_keywords = exclusion_config.get('tax_description_keywords', ['tax', 'gst input tax', 'sales tax'])
    for kw in tax_keywords:
        if kw in line_desc:
            return 0, 'NON_PROCUREMENT'
            
    # Check status if column exists
    status = str(row.get('status', '')).strip().lower()
    if status == 'cancelled':
        return 0, 'CANCELLED'
    if status in ('rejected', 'on hold', 'disputed'):
        return 0, 'NOT_APPROVED'

    return 1, ''
