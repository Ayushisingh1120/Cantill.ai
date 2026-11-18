import re
import pandas as pd
import numpy as np
from datetime import datetime

def clean_amount(val) -> float:
    """Parse amounts with commas, quotes, symbols into clean floats."""
    if pd.isna(val):
        return 0.0
    if isinstance(val, (int, float, np.integer, np.floating)):
        return float(val)
    s = str(val).replace(',', '').replace(' ', '').replace('$', '').strip()
    try:
        return float(s)
    except Exception:
        return 0.0

def parse_vat_rate(val, default_rate: float = 0.15) -> float:
    """Parse VAT rates like '15%', 15, 0.15, 0, NaN."""
    if pd.isna(val) or val is None or str(val).strip() == '':
        return default_rate
    s = str(val).replace('%', '').strip()
    try:
        f = float(s)
        if f > 1.0:
            return f / 100.0
        return f
    except Exception:
        return default_rate

def remove_vat(gross_amount: float, vat_rate: float) -> float:
    """Calculate net amount from gross amount and VAT rate."""
    return gross_amount / (1.0 + vat_rate)

def parse_date(val):
    """
    Parse varied date formats: YYYY-MM-DD, DD/MM/YYYY, DD-Mon-YY, etc.
    Returns ISO string 'YYYY-MM-DD' or None if unparseable.
    """
    if pd.isna(val) or val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in ('nan', 'none', 'nat'):
        return None
    
    # Try common formats explicitly first for speed
    for fmt in ('%Y-%m-%d', '%d/%m/%Y', '%m/%d/%Y', '%d-%b-%y', '%d-%b-%Y', '%Y/%m/%d', '%d.%m.%Y'):
        try:
            dt = datetime.strptime(s, fmt)
            # Basic sanity check on year
            if 1990 <= dt.year <= 2040:
                return dt.strftime('%Y-%m-%d')
            return None
        except ValueError:
            continue
    
    # Fallback to pandas
    try:
        dt = pd.to_datetime(s, errors='coerce')
        if pd.notna(dt) and 1990 <= dt.year <= 2040:
            return dt.strftime('%Y-%m-%d')
    except Exception:
        pass
    return None

class CurrencyConverter:
    """
    Handles reporting currency conversion:
    - Direct line rate if present (units per 1 USD)
    - Monthly FX table lookup
    - Nearest month fallback if month is not in table
    """
    def __init__(self, fx_file_path=None, reporting_currency="USD"):
        self.reporting_currency = reporting_currency
        self.fx_table = {}
        self.currency_months = {}
        if fx_file_path:
            self._load_fx_table(fx_file_path)

    def _load_fx_table(self, fx_file_path):
        try:
            df = pd.read_csv(fx_file_path)
            for _, r in df.iterrows():
                curr = str(r['currency']).strip().upper()
                month = str(r['month']).strip()
                rate = float(r['units_per_usd'])
                self.fx_table[(curr, month)] = rate
                if curr not in self.currency_months:
                    self.currency_months[curr] = []
                self.currency_months[curr].append((month, rate))
            # Sort months for each currency
            for curr in self.currency_months:
                self.currency_months[curr].sort(key=lambda x: x[0])
        except Exception as e:
            print(f"Warning: Failed to load FX table: {e}")

    def convert_to_reporting(self, amount: float, currency: str, line_rate=None, date_str=None) -> float:
        if pd.isna(amount) or amount == 0.0:
            return 0.0
        curr = str(currency).strip().upper() if pd.notna(currency) else self.reporting_currency
        if curr == self.reporting_currency:
            return float(amount)
        
        # 1. Use line_rate if available and > 0
        if pd.notna(line_rate) and float(line_rate) > 0:
            return float(amount) / float(line_rate)
        
        # 2. Lookup in FX table using month
        month = None
        if date_str and len(str(date_str)) >= 7:
            month = str(date_str)[:7]
        
        if month and (curr, month) in self.fx_table:
            return float(amount) / self.fx_table[(curr, month)]
        
        # 3. Fallback to nearest month for that currency
        if curr in self.currency_months and self.currency_months[curr]:
            available = self.currency_months[curr]
            if month:
                # find closest month lexicographically
                closest_rate = min(available, key=lambda x: abs(int(x[0].replace('-', '')) - int(month.replace('-', ''))))[1]
            else:
                closest_rate = available[-1][1]
            return float(amount) / closest_rate

        # If no rate found at all, return amount as-is
        return float(amount)

def clean_text(text: str) -> str:
    """Clean and normalize OCR/raw item and supplier descriptions."""
    if pd.isna(text) or text is None:
        return ""
    s = str(text)
    # Replace non-breaking spaces and formatting artifacts
    s = s.replace('\xa0', ' ').replace('\n', ' ').replace('\r', ' ')
    # Normalize excessive punctuation but keep part number symbols
    s = re.sub(r'[\t\r\n]+', ' ', s)
    s = re.sub(r'\s+', ' ', s)
    return s.strip()
