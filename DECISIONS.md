# DECISIONS.md — Spend Pipeline & Classifier Design Decisions

This document records the empirical discoveries from exploratory data analysis across Company A and Company B, and the architectural and algorithmic decisions made to resolve every ambiguity, edge case, and system discrepancy.

---

## 1. Company A: Multi-System Reconciliation & De-duplication

### Finding 1.1: System Hierarchy and Overlap
* **Data Reality**: Company A operates two systems:
  * `erp_goods_receipts.csv` (37,029 lines): SAP goods receipts with PO numbers, line items, material IDs, vendor IDs, and net value in ZAR.
  * `invoice_tool_export.csv` (28,520 lines): Accounts-payable tool export with OCR'd invoice line items, gross amounts, VAT rates, and PO references.
  * Over 17,600 tool lines explicitly reference PO numbers that exist in the ERP goods receipt records.
* **Decision**: 
  * The ERP system represents physical goods/services received and validated by operations. It is designated as the **primary ground-truth system**. All 37,029 ERP lines are included as spend (`include = 1`).
  * Any line in the invoice tool that matches an ERP record by purchase order number is excluded with code `DUPLICATE` to prevent double-counting.
  * Any line in the invoice tool that does **not** match an ERP record represents direct AP spend (services, off-PO spend, or receipts not yet loaded into ERP) and is preserved as real spend if approved/paid.

### Finding 1.2: Gross vs. Net Tax Stripping
* **Data Reality**: ERP amounts (`value_zar`) are already net of tax. Tool amounts (`gross_amount`) are gross and include South African VAT. The `vat_rate` field contains mixed types: numeric `0.15`, integer `15`, string `"15%"`, zero `0`, and empty/null values.
* **Decision**:
  * Standardized VAT rate parser converts `"15%"`, `15`, `0.15` to float `0.15`.
  * As instructed by the task specification, blank or null VAT rates default strictly to South African statutory VAT of `15%` (`0.15`).
  * Net spend is calculated as:
    $$\text{spend}_{\text{net}} = \frac{\text{gross\_amount}}{1.0 + \text{vat\_rate}}$$
  * Negative amounts (e.g. credit notes or return goods receipts) preserve their sign to accurately net out refunds in the spend cube.

### Finding 1.3: Tool Line Status and Internal Duplicates
* **Data Reality**: 
  * Status in the tool export has 5 distinct values: `Paid` (19,837), `Approved` (8,180), `Cancelled` (188), `On Hold` (172), `Rejected` (143).
  * 562 exact duplicate lines exist within `invoice_tool_export.csv` where an invoice was exported multiple times with identical line text and gross amount.
* **Decision**:
  * Status `Cancelled` $\rightarrow$ `include = 0, exclusion_reason = CANCELLED`.
  * Status `On Hold` and `Rejected` $\rightarrow$ `include = 0, exclusion_reason = NOT_APPROVED`.
  * Repeated exports of the same invoice are deduplicated: the first occurrence is evaluated, and subsequent identical lines are excluded with `exclusion_reason = DUPLICATE`.

---

## 2. Company B: Multi-Currency & Non-Procurement Exclusions

### Finding 2.1: Non-Procurement Identification
* **Data Reality**: Analysis of Company B's 3,000 seed labels showed 450 excluded lines, all tagged as `NON_PROCUREMENT`.
  * Cross-referencing with general ledger accounts revealed that every GL account description starting with `IC` or `I/C` (e.g. `IC06`, `IC Payables`, `I/C Payable`, `IC05`, `IC67`) represents Intercompany transfers.
  * Line descriptions containing `"Sales Tax Expense"`, `"Sales Tax"`, `"GST Input Tax"`, or `"State & Local Taxes"` represent tax accruals, not third-party supplier procurement spend.
* **Decision**:
  * Any line where `gl_account_description` matches `^IC` or `^I/C` is excluded as `NON_PROCUREMENT`.
  * Any line where `line_description` contains tax keywords (`"tax"`, `"sales tax"`, `"gst input tax"`, `"state & local taxes"`) is excluded as `NON_PROCUREMENT`.
  * Excluded lines have `spend = 0.0` and empty category fields.

### Finding 2.2: Multi-Currency Conversion
* **Data Reality**: Company B features 25 currencies across 8 global operating regions (JDE and SAP).
  * Line-level `currency_rate` is populated for 19,726 non-USD lines (expressed in units of local currency per 1 USD).
  * 3,343 non-USD lines lack a line rate and require external FX rates from `fx_rates_usd.csv`.
  * 5 historical lines in 2022 predated the earliest monthly rate in the table.
* **Decision**:
  * USD lines are taken as-is: $\text{spend}_{\text{USD}} = \text{line\_amount}$.
  * Non-USD lines with valid `currency_rate` use line conversion:
    $$\text{spend}_{\text{USD}} = \frac{\text{line\_amount}}{\text{currency\_rate}}$$
  * Non-USD lines lacking `currency_rate` look up $(currency, month)$ in `fx_rates_usd.csv`.
  * Where the exact month is missing (e.g. historical 2022 CAD lines), the converter falls back to the nearest chronological month for that currency.

---

## 3. Entity Resolution: Supplier and Item De-duplication

### Finding 3.1: Supplier Resolution
* **Data Reality**: 
  * In Company A, the golden supplier labels (`labels_A.csv`) demonstrate that vendor IDs map 1-to-1 with golden supplier keys (`Adjusted Rand Index > 0.999` for named vendors). Long-tail one-off vendors (missing vendor IDs) were grouped under a sundry supplier cluster (`SG-f85711`).
  * In Company B, supplier names were omitted by design, with only `vendor_id` provided.
* **Decision**:
  * A two-tier resolution strategy was implemented:
    1. If `vendor_id` is present, it anchors the supplier cluster.
    2. If `vendor_id` is missing, the supplier name is normalized: legal entities (`(PTY) LTD`, `LTD`, `CC`, `BV`, `GMBH`, `INC`, `CORP`) and non-alphanumeric punctuation are removed, whitespace collapsed, and names clustered.

### Finding 3.2: Item Resolution
* **Data Reality**: Material numbers (`material_number`) map directly to golden item keys in `labels_A.csv` (`NMI = 0.9677`). Service lines and free-text purchases without material numbers had null item keys in the golden set.
* **Decision**:
  * Where a structured material/item ID is present, it anchors the `item_key`.
  * Where material ID is absent, an item cluster key is deterministically generated from the normalized description and supplier identifier, ensuring identical items bought repeatedly receive the same key.

---

## 4. Unseen Taxonomy Classification

### Finding 4.1: The Zero-Shot Challenge
* **Data Reality**: Company C at the blind test has a completely unseen category tree (up to 4 levels) with zero labeled examples. Company A's labels must also be capable of being completely hidden.
* **Decision**:
  * Designed a **Dual-Mode Hierarchical Classifier**:
    1. **Zero-Shot Mode**: Embeds full category paths (`L1 > L2 > L3 > L4`) using a local SentenceTransformer (`all-MiniLM-L6-v2`) combined with token n-gram TF-IDF lexical overlap. Operates 100% offline with zero external API calls.
    2. **Supervised / Transfer Mode**: When seed labels (Company B) or golden labels (Company A) are provided, the pipeline trains a calibrated linear SGD classifier that routes high-confidence matches, falling back to zero-shot semantic matching for unseen tokens.
  * No category names are hardcoded in the codebase; the taxonomy is ingested dynamically at runtime.
