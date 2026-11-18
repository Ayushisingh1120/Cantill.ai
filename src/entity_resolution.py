import re
import hashlib
import pandas as pd
from typing import Optional

def normalize_supplier_name(name: Optional[str]) -> str:
    """
    Cleans legal entities, punctuation, whitespace to resolve real suppliers.
    E.g. 'KINORA INDUSTRIAL SUPPLIES ENGINEERING (PTY) LTD' -> 'kinora industrial supplies engineering'
    """
    if pd.isna(name) or name is None:
        return ""
    s = str(name).lower()
    # Strip legal suffixes
    s = re.sub(r'\(pty\)\s*ltd\.?|\bpty\s*ltd\.?|\bltd\.?|\bcc\b|\bbv\b|\bgmbh\b|\bsa\b|\binc\b|\bcorp\b|\bco\b', '', s)
    # Strip non-alphanumeric except spaces
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def generate_cluster_key(prefix: str, text: str) -> str:
    """Generate a stable, clean key for a group/cluster."""
    h = hashlib.md5(text.encode('utf-8')).hexdigest()[:8]
    return f"{prefix}-{h}"

class EntityResolver:
    """
    Resolves supplier_key and item_key across lines and sources.
    """
    def __init__(self):
        self.supplier_map = {}
        self.item_map = {}

    def resolve_supplier(self, vendor_id: Optional[str], supplier_name: Optional[str]) -> str:
        # Check vendor_id first
        v_id = str(vendor_id).strip() if pd.notna(vendor_id) and str(vendor_id).strip() not in ('', 'nan', 'None') else None
        norm_name = normalize_supplier_name(supplier_name)
        
        if v_id:
            # Map vendor ID to consistent supplier key
            if v_id not in self.supplier_map:
                key_text = norm_name if norm_name else v_id
                self.supplier_map[v_id] = generate_cluster_key("SUP", key_text)
            return self.supplier_map[v_id]
        
        if norm_name:
            if norm_name not in self.supplier_map:
                self.supplier_map[norm_name] = generate_cluster_key("SUP", norm_name)
            return self.supplier_map[norm_name]
            
        return "SUP-MISC"

    def resolve_item(self, material_num: Optional[str], description: Optional[str], supplier_key: str = "") -> str:
        m_num = str(material_num).strip() if pd.notna(material_num) and str(material_num).strip() not in ('', 'nan', 'None') else None
        if m_num:
            if m_num not in self.item_map:
                self.item_map[m_num] = generate_cluster_key("ITEM", m_num)
            return self.item_map[m_num]
            
        # Fallback to normalized description
        desc = re.sub(r'[^a-z0-9\s]', ' ', str(description).lower()).strip()
        desc = re.sub(r'\s+', ' ', desc)
        if desc and len(desc) > 3:
            key_source = f"{desc[:50]}_{supplier_key}"
            if key_source not in self.item_map:
                self.item_map[key_source] = generate_cluster_key("ITEM", key_source)
            return self.item_map[key_source]
            
        return "ITEM-UNSPECIFIED"
