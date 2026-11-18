# Cantill.ai Spend Data Pipeline & Zero-Shot Classifier

A production-grade, configuration-driven spend data intelligence pipeline and classifier built for enterprise procurement data reconciliation, currency normalization, entity resolution, and zero-shot taxonomy classification.

---

## 1. Quick Start: The One Command

To run the complete pipeline end-to-end on both companies (Company A and Company B), generating the validated output files:

```bash
python run_pipeline.py
```

This single command:
1. Ingests Company A (`erp_goods_receipts.csv` and `invoice_tool_export.csv`), de-duplicates across systems, cleans VAT, resolves entities, classifies lines, and writes `Ayushisingh1120__A.csv`.
2. Ingests Company B (`invoice_lines.csv`), converts 25 currencies to USD, filters non-procurement spend, resolves entities, classifies lines, and writes `Ayushisingh1120__B.csv`.
3. Completes in **~4.3 minutes** for all 106,577 lines on standard CPU (14x faster than the 60-minute benchmark).

---

## 2. Environment Setup

```bash
# Clone the repository
git clone <your-repo-link>
cd cantill_spend_tool

# Install required dependencies
pip install pandas numpy scikit-learn sentence-transformers pyyaml
```

The tool uses a local sentence transformer model (`all-MiniLM-L6-v2`) and local linear classifiers. It runs **100% offline with zero external API calls or API keys**.

---

## 3. Running for Individual Companies & Blind Test (Company C)

### Run Company A:
```bash
python run_pipeline.py --config configs/company_A.yaml
```

### Run Company B:
```bash
python run_pipeline.py --config configs/company_B.yaml
```

### Run with Labels Hidden (Zero-Shot Mode):
To evaluate the tool on Company A with labels hidden (classifying purely from the category tree alone):
```bash
python run_pipeline.py --config configs/company_A.yaml --zero-shot
```

### Blind Test Execution (Company C):
When Company C is provided during the blind test:
1. Copy or edit `configs/company_C_template.yaml` to point to Company C's input file and taxonomy file.
2. Run:
```bash
python run_pipeline.py --config configs/company_C.yaml --user Ayushisingh1120 --output-dir submission
```
This produces `submission/Ayushisingh1120__C.csv`.

---

## 4. Repository Structure

```
cantill_spend_tool/
├── configs/
│   ├── company_A.yaml            # Config for Company A (multi-system ERP + AP tool)
│   ├── company_B.yaml            # Config for Company B (multi-currency, multi-ERP)
│   └── company_C_template.yaml   # Ready template for Company C blind test
├── src/
│   ├── clean.py                  # Amount cleaning, date parsing, VAT stripping, FX converter
│   ├── exclusions.py             # Exclusion logic (DUPLICATE, CANCELLED, NOT_APPROVED, NON_PROCUREMENT)
│   ├── entity_resolution.py      # Supplier and item clustering and normalization
│   ├── classifier.py             # Zero-shot semantic + lexical hybrid hierarchical classifier
│   └── pipeline.py               # Pipeline orchestrator
├── run_pipeline.py               # Single-command CLI entrypoint
├── Ayushisingh1120__A.csv         # Validated output for Company A (65,549 lines)
├── Ayushisingh1120__B.csv         # Validated output for Company B (41,028 lines)
├── APPROACH.md                   # One-page technical methodology, accuracy & runtime benchmarks
├── DECISIONS.md                  # Comprehensive record of empirical data findings & decisions
├── README.md                     # Setup and execution guide
└── .gitignore                    # Ensures task data files are never committed
```

---

## 5. Output Format

Both output files strictly follow the mandated format:
`record_id,include,exclusion_reason,supplier_key,item_key,spend,l1,l2,l3,l4,confidence`

* **For Company A**: Includes all records from both files (`ERP:<line_id>` and `TOOL:<tool_record_id>`).
* **For Company B & C**: Contains `<line_id>`.
* **Included lines** (`include = 1`): Net spend in reporting currency, lowest-level taxonomy classification, and confidence.
* **Excluded lines** (`include = 0`): Reason code (`DUPLICATE`, `NOT_APPROVED`, `CANCELLED`, `NON_PROCUREMENT`), `spend = 0.0`, empty category levels, and confidence.
