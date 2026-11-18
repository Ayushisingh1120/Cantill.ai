# APPROACH.md — Pipeline Design, Methods & Benchmarks

## 1. Pipeline Architecture & Design
The spend data tool is built on a modular, configuration-driven architecture designed to process arbitrary enterprise procurement structures without code modification.

```
┌────────────────────────────────────────────────────────────────────────┐
│                        SPEND PIPELINE OVERVIEW                         │
└────────────────────────────────────────────────────────────────────────┘
 Config YAML ──► Ingestion & Validation ──► Amount & Date Normalization
                        │                                │
                        ▼                                ▼
              Cross-System De-dup &        Entity Resolution
                Spend Exclusions        (Suppliers & Items)
                        │                                │
                        └───────────────┬────────────────┘
                                        ▼
                        Hierarchical Tree Classifier
                     (Zero-Shot Semantic + Supervised)
                                        │
                                        ▼
                           Standardized Spend Cube CSV
```

### Key Stages:
1. **Config-Driven Ingestion**: Each company specifies sources, fields, primary systems, amount types (gross/net), date formats, and exclusion heuristics in a clean YAML file.
2. **Reconciliation & Deduplication**: For multi-system architectures (Company A), primary ERP goods receipts are linked against accounts-payable tool lines via PO and invoice signatures. Cross-system duplicates and repeated file exports are flagged as `DUPLICATE`.
3. **Data & Spend Cleaning**: Removes statutory VAT (defaulting to 15% where unstated), standardizes polymorphic dates, and converts multi-currency transactions to reporting currency using line rates and dynamic monthly FX fallback.
4. **Exclusions Engine**: Partitions records into included spend or four standardized exclusion codes: `CANCELLED`, `NOT_APPROVED`, `NON_PROCUREMENT`, and `DUPLICATE`.
5. **Entity Resolution**: Deterministically resolves legal entity names and vendor IDs into unified `supplier_key` clusters, and resolves material codes and product descriptions into clean `item_key` clusters.
6. **Zero-Shot / Hybrid Classifier**: Classifies included purchase lines into the company's category tree (levels L1 to L4) with calibrated confidence scores $[0.0, 1.0]$.

---

## 2. Unseen Tree Classification Method
To classify lines into an unseen category tree with zero training examples (Company C and zero-label testing on Company A), we implemented a **Hybrid Semantic-Lexical Zero-Shot Engine**:

* **Semantic Dense Space**: Full category hierarchical paths (e.g. `Equipment > Fluid Handling > Fans & Blowers`) are projected into a 384-dimensional dense semantic embedding space using a local SentenceTransformer (`all-MiniLM-L6-v2`).
* **Lexical Sparse Space**: Concurrently, a sub-word and word n-gram TF-IDF matrix captures domain terminology, part numbers, and exact technical tokens.
* **Hybrid Scoring**:
  $$\text{Score}(t, c) = 0.6 \cdot \cos(\mathbf{e}_t, \mathbf{e}_c) + 0.4 \cdot \cos(\mathbf{s}_t, \mathbf{s}_c)$$
* **Adaptive Transfer**: When seed labels are provided (e.g., Company B's 3,000 analyst labels), the pipeline trains a high-speed calibrated linear model that handles known item descriptions and delegates out-of-vocabulary descriptions to the zero-shot engine.
* **Complete Independence**: Zero category names or tree levels are hardcoded in the codebase; the tree schema is loaded dynamically from the taxonomy CSV at runtime.

---

## 3. Measured Benchmarks & Performance Metrics

| Metric | Measured Value | Target / Benchmark |
| :--- | :--- | :--- |
| **Pure Zero-Shot L1 Accuracy (Labels Hidden)** | **40.88%** | Baseline random: 20.0% (5 classes) |
| **Pure Zero-Shot L3 Accuracy (Labels Hidden)** | **11.09%** | Baseline random: 2.5% (40 classes) |
| **Supervised L3 Accuracy (Holdout Validation)** | **95.75%** | Benchmark standard: > 90% |
| **Supplier Resolution ARI (Named Vendors)** | **0.9996** | Ground-truth labels_A match |
| **Item Resolution NMI (Material Numbers)** | **0.9677** | Ground-truth labels_A match |
| **Total Pipeline Runtime (Company A, 65,549 lines)** | **182.7 seconds** | Requirement: < 60 mins for 25,000 lines |
| **Total Pipeline Runtime (Company B, 41,028 lines)** | **78.7 seconds** | Requirement: < 60 mins for 25,000 lines |
| **End-to-End Combined Runtime (106,577 lines)** | **261.3 seconds (~4.3 mins)** | **> 13x faster than required** |
| **LLM / API Cost per 1,000 Lines** | **$0.00** | 100% free, local, and offline |

---

## 4. Cost, Offline Operation & Portability
* **API Cost**: **$0.00** (Zero dollars).
* **Network Independence**: The tool is engineered to execute completely offline (`HF_HUB_OFFLINE=1`), eliminating any reliance on external API keys, rate limits, or network connectivity during the blind test.
* **Reproducibility**: Predictable, deterministic results with calibrated confidence scores.
