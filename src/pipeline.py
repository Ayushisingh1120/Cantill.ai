import os
import yaml
import pandas as pd
import numpy as np
from typing import Dict, Any, List

from src.clean import (
    clean_amount,
    parse_vat_rate,
    remove_vat,
    parse_date,
    clean_text,
    CurrencyConverter
)
from src.exclusions import (
    extract_po_number,
    check_exclusion_company_a_tool,
    check_exclusion_company_b
)
from src.entity_resolution import EntityResolver
from src.classifier import SpendClassifier

class SpendPipeline:
    def __init__(self, config_path: str, use_zero_shot: bool = False, base_dir: str = "."):
        self.base_dir = base_dir
        self.use_zero_shot = use_zero_shot
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = yaml.safe_load(f)
            
        self.company_name = self.config['company_name']
        self.reporting_currency = self.config.get('reporting_currency', 'USD')
        
        # Load taxonomy
        tax_path = os.path.join(self.base_dir, self.config['taxonomy_file'])
        self.taxonomy_df = pd.read_csv(tax_path)
        self.classifier = SpendClassifier(self.taxonomy_df, use_zero_shot=self.use_zero_shot)
        
        # FX converter if present
        fx_file = self.config.get('fx_file')
        fx_path = os.path.join(self.base_dir, fx_file) if fx_file else None
        self.fx_converter = CurrencyConverter(fx_path, reporting_currency=self.reporting_currency)
        
        self.entity_resolver = EntityResolver()

    def train_classifier_if_available(self):
        if self.use_zero_shot:
            return
            
        # Check for Company A labels
        labels_a_path = os.path.join(self.base_dir, "Cantill_DataML_Task_Kit/company_A/labels_A.csv")
        erp_path = os.path.join(self.base_dir, "Cantill_DataML_Task_Kit/company_A/erp_goods_receipts.csv")
        if self.company_name == 'company_A' and os.path.exists(labels_a_path) and os.path.exists(erp_path):
            print("Training classifier on Company A labels...")
            df_lbl = pd.read_csv(labels_a_path)
            df_erp = pd.read_csv(erp_path, low_memory=False)
            merged = df_lbl.merge(df_erp, left_on='record_id', right_on=df_erp['line_id'].apply(lambda x: f"ERP:{x}"))
            texts = (merged['material_short_text'].fillna('') + ' ' + 
                     merged['service_short_text'].fillna('') + ' ' + 
                     merged['supplier_name'].fillna('')).tolist()
            labels = merged['category_l3'].tolist()
            self.classifier.train_supervised(texts, labels)
            return

        # Check for Company B seed labels
        seed_b_path = os.path.join(self.base_dir, "Cantill_DataML_Task_Kit/company_B/seed_labels.csv")
        inv_b_path = os.path.join(self.base_dir, "Cantill_DataML_Task_Kit/company_B/invoice_lines.csv")
        if self.company_name == 'company_B' and os.path.exists(seed_b_path) and os.path.exists(inv_b_path):
            print("Training classifier on Company B seed labels...")
            df_seed = pd.read_csv(seed_b_path)
            df_inv = pd.read_csv(inv_b_path, low_memory=False)
            # Only train on included lines with non-null labels
            valid_seed = df_seed[(df_seed['include'] == 1) & (df_seed['category_l3'].notna())]
            merged = valid_seed.merge(df_inv, on='line_id')
            texts = (merged['line_description'].fillna('') + ' ' + 
                     merged['gl_account_description'].fillna('')).tolist()
            labels = merged['category_l3'].tolist()
            self.classifier.train_supervised(texts, labels)

    def run(self) -> pd.DataFrame:
        print(f"Running pipeline for {self.company_name} (Zero-shot: {self.use_zero_shot})...")
        self.train_classifier_if_available()
        
        all_records = []
        
        if self.company_name == 'company_A':
            records = self._process_company_a()
            all_records.extend(records)
        elif self.company_name == 'company_B':
            records = self._process_company_b()
            all_records.extend(records)
        else:
            records = self._process_generic_company()
            all_records.extend(records)
            
        df_out = pd.DataFrame(all_records)
        
        # Classify included lines
        inc_mask = df_out['include'] == 1
        inc_indices = df_out[inc_mask].index
        
        print(f"Classifying {len(inc_indices)} included lines into taxonomy...")
        inc_texts = df_out.loc[inc_indices, '_text_for_classification'].tolist()
        predictions = self.classifier.predict_batch(inc_texts)
        
        for idx, (path_dict, conf) in zip(inc_indices, predictions):
            df_out.at[idx, 'l1'] = path_dict.get('l1', '')
            df_out.at[idx, 'l2'] = path_dict.get('l2', '')
            df_out.at[idx, 'l3'] = path_dict.get('l3', '')
            df_out.at[idx, 'l4'] = path_dict.get('l4', '')
            df_out.at[idx, 'confidence'] = conf
            
        # For excluded lines, assign confidence from text matching, leave categories blank
        ex_mask = df_out['include'] == 0
        ex_indices = df_out[ex_mask].index
        if len(ex_indices) > 0:
            ex_texts = df_out.loc[ex_indices, '_text_for_classification'].tolist()
            ex_preds = self.classifier.predict_batch(ex_texts)
            for idx, (_, conf) in zip(ex_indices, ex_preds):
                df_out.at[idx, 'confidence'] = conf
                df_out.at[idx, 'l1'] = ''
                df_out.at[idx, 'l2'] = ''
                df_out.at[idx, 'l3'] = ''
                df_out.at[idx, 'l4'] = ''
                df_out.at[idx, 'spend'] = 0.0

        # Ensure correct column ordering and formats
        output_cols = [
            'record_id', 'include', 'exclusion_reason', 
            'supplier_key', 'item_key', 'spend', 
            'l1', 'l2', 'l3', 'l4', 'confidence'
        ]
        
        # Round spend to 2 decimal places
        df_out['spend'] = df_out['spend'].apply(lambda x: round(float(x), 2) if pd.notna(x) else 0.0)
        df_out['confidence'] = df_out['confidence'].apply(lambda x: round(float(x), 2) if pd.notna(x) else 0.50)
        
        final_df = df_out[output_cols].copy()
        print(f"Pipeline finished for {self.company_name}. Output shape: {final_df.shape}")
        return final_df

    def _process_company_a(self) -> List[Dict[str, Any]]:
        erp_source = next(s for s in self.config['sources'] if s['name'] == 'erp')
        tool_source = next(s for s in self.config['sources'] if s['name'] == 'invoice_tool')
        
        df_erp = pd.read_csv(os.path.join(self.base_dir, erp_source['file']), low_memory=False)
        df_tool = pd.read_csv(os.path.join(self.base_dir, tool_source['file']), low_memory=False)
        
        records = []
        erp_pos = set()
        
        # 1. Process ERP Goods Receipts (Primary source of spend)
        for _, r in df_erp.iterrows():
            line_id = str(r[erp_source['id_col']]).strip()
            rec_id = f"ERP:{line_id}"
            val_zar = clean_amount(r[erp_source['amount_col']])
            
            # Record PO
            po = extract_po_number(r.get(erp_source['po_number_col']))
            if po:
                erp_pos.add(po)
                
            sup_key = self.entity_resolver.resolve_supplier(
                r.get(erp_source['vendor_id_col']),
                r.get(erp_source['supplier_col'])
            )
            item_key = self.entity_resolver.resolve_item(
                r.get(erp_source['item_id_col']),
                str(r.get('material_short_text', '')) + " " + str(r.get('service_short_text', '')),
                sup_key
            )
            
            desc_text = f"{r.get('material_short_text', '')} {r.get('service_short_text', '')} {r.get(erp_source['supplier_col'], '')}"
            
            records.append({
                'record_id': rec_id,
                'include': 1,
                'exclusion_reason': '',
                'supplier_key': sup_key,
                'item_key': item_key,
                'spend': val_zar,
                '_text_for_classification': desc_text,
                'l1': '', 'l2': '', 'l3': '', 'l4': '',
                'confidence': 0.85
            })
            
        # 2. Process Invoice Tool Export (Secondary)
        seen_tool_signatures = set()
        status_rules = tool_source.get('status_rules', {})
        
        for _, r in df_tool.iterrows():
            t_id = str(r[tool_source['id_col']]).strip()
            rec_id = f"TOOL:{t_id}"
            
            gross = clean_amount(r[tool_source['amount_col']])
            vat = parse_vat_rate(r.get(tool_source['vat_col']), default_rate=0.15)
            net_spend = remove_vat(gross, vat)
            
            r_dict = r.to_dict()
            r_dict['clean_gross'] = gross
            inc, exc_reason = check_exclusion_company_a_tool(
                pd.Series(r_dict), erp_pos, seen_tool_signatures, status_rules
            )
            
            sup_name = r.get(tool_source['supplier_col'])
            sup_key = self.entity_resolver.resolve_supplier(None, sup_name)
            item_key = self.entity_resolver.resolve_item(None, r.get('line_text'), sup_key)
            
            desc_text = f"{r.get('line_text', '')} {sup_name}"
            spend_val = net_spend if inc == 1 else 0.0
            
            records.append({
                'record_id': rec_id,
                'include': inc,
                'exclusion_reason': exc_reason,
                'supplier_key': sup_key,
                'item_key': item_key,
                'spend': spend_val,
                '_text_for_classification': desc_text,
                'l1': '', 'l2': '', 'l3': '', 'l4': '',
                'confidence': 0.80
            })
            
        return records

    def _process_company_b(self) -> List[Dict[str, Any]]:
        src = self.config['sources'][0]
        df_inv = pd.read_csv(os.path.join(self.base_dir, src['file']), low_memory=False)
        exclusion_cfg = self.config.get('exclusion_rules', {}).get('non_procurement', {})
        
        records = []
        for _, r in df_inv.iterrows():
            line_id = str(r[src['id_col']]).strip()
            amt = clean_amount(r[src['amount_col']])
            curr = str(r.get(src['currency_col'], 'USD')).strip()
            line_rate = r.get(src['currency_rate_col'])
            date_str = parse_date(r.get(src['date_col']))
            
            # Convert to USD
            spend_usd = self.fx_converter.convert_to_reporting(
                amt, curr, line_rate=line_rate, date_str=date_str
            )
            
            inc, exc_reason = check_exclusion_company_b(r, exclusion_cfg)
            
            vendor_id = r.get(src['vendor_id_col'])
            sup_key = self.entity_resolver.resolve_supplier(vendor_id, None)
            
            item_num = r.get(src['item_id_col'])
            desc = r.get('line_description')
            item_key = self.entity_resolver.resolve_item(item_num, desc, sup_key)
            
            spend_val = spend_usd if inc == 1 else 0.0
            desc_text = f"{desc or ''} {r.get('gl_account_description', '')}"
            
            records.append({
                'record_id': line_id,
                'include': inc,
                'exclusion_reason': exc_reason,
                'supplier_key': sup_key,
                'item_key': item_key,
                'spend': spend_val,
                '_text_for_classification': desc_text,
                'l1': '', 'l2': '', 'l3': '', 'l4': '',
                'confidence': 0.80
            })
            
        return records

    def _process_generic_company(self) -> List[Dict[str, Any]]:
        src = self.config['sources'][0]
        df = pd.read_csv(os.path.join(self.base_dir, src['file']), low_memory=False)
        records = []
        
        for _, r in df.iterrows():
            line_id = str(r[src['id_col']]).strip()
            amt = clean_amount(r[src['amount_col']])
            if src.get('amount_type') == 'gross':
                vat = parse_vat_rate(r.get(src.get('vat_col')), default_rate=src.get('default_vat_rate', 0.15))
                amt = remove_vat(amt, vat)
                
            curr = str(r.get(src.get('currency_col', 'currency'), self.reporting_currency)).strip()
            rate = r.get(src.get('currency_rate_col'))
            date_str = parse_date(r.get(src.get('date_col', 'invoice_date')))
            
            spend_rep = self.fx_converter.convert_to_reporting(amt, curr, line_rate=rate, date_str=date_str)
            
            status = str(r.get(src.get('status_col', 'status'), '')).strip().lower()
            inc = 1
            exc_reason = ""
            if status == 'cancelled':
                inc, exc_reason = 0, 'CANCELLED'
            elif status in ('rejected', 'on hold', 'disputed'):
                inc, exc_reason = 0, 'NOT_APPROVED'
                
            sup_name = r.get(src.get('supplier_col'))
            v_id = r.get(src.get('vendor_id_col'))
            sup_key = self.entity_resolver.resolve_supplier(v_id, sup_name)
            
            item_num = r.get(src.get('item_id_col'))
            desc_cols = src.get('description_cols', ['line_description'])
            desc = " ".join([str(r.get(c, '')) for c in desc_cols if pd.notna(r.get(c))])
            item_key = self.entity_resolver.resolve_item(item_num, desc, sup_key)
            
            records.append({
                'record_id': line_id,
                'include': inc,
                'exclusion_reason': exc_reason,
                'supplier_key': sup_key,
                'item_key': item_key,
                'spend': spend_rep if inc == 1 else 0.0,
                '_text_for_classification': desc,
                'l1': '', 'l2': '', 'l3': '', 'l4': '',
                'confidence': 0.80
            })
            
        return records
