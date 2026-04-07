# Zap Product Deduplication Engine

> **GenAI Exploration Lead - Home Assignment**  

---

## Approach

### 1. The Problem
A product list contains duplicates with inconsistent names (Hebrew/English mixed, different word order, extra descriptors).  
The goal is to merge duplicates and surface the lowest available price.

### 2. Solution - 4-Step Pipeline

1. **Normalization (`utils.py`)**  
   - Hebrew brands → English (`"סמסונג" → "samsung"`)  
   - Remove punctuation and noise words  
   - Collapse whitespace  

2. **TF-IDF Blocking (`deduplicator.py`)**  
   - Character n-gram cosine similarity  
   - Compare only within the same category  
   - Produce a small set of candidates (avoids O(n²) full comparison)  

3. **Rule-based Verification**  
   - Storage mismatch → NOT duplicate (`"128GB" ≠ "256GB"`)  
   - Screen size mismatch → NOT duplicate (`"55 inch" ≠ "65 inch"`)  
   - Re-check cosine similarity with tighter threshold  

4. **Union-Find Grouping + Aggregation**  
   - Merge confirmed duplicate pairs  
   - Handle chains: A=B, B=C → A=B=C  
   - Surface the lowest price per group  

---

## Project Structure

zap-dedup/
├── data/
│   └── products_sample.csv   # 20 sample products with deliberate duplicates
├── src/
│   ├── deduplicator.py       # Main pipeline
│   └── utils.py              # Normalization, Hebrew→English map, helpers
├── tests/
│   └── test_deduplicator.py  # pytest - covers normalize, rules, full pipeline
├── requirements.txt
└── README.md

---

## Quick Start

# 1. Install dependencies
pip install -r requirements.txt

# 2. Run on sample data
cd src
python deduplicator.py ../data/products_sample.csv --output results.csv

# 3. Run tests
pytest ../tests/

### Example Output

[1/5] Loaded 20 products.
[2/5] 41 candidate pairs.
[3/5] 21 duplicate pairs confirmed.
[4/5] Done: 20 → 7 groups.
GROUPS:

  Samsung S23 256GB  (4 items)
     Lowest: ₪2,850 @ ivory
    - ₪2,999  Samsung Galaxy S23
    - ₪3,100  סמסונג גלקסי S23
    - ₪2,850  Samsung S23 256GB
    - ₪3,200  galaxy s23 samsung

  Apple iPhone 14 Pro  (3 items)
     Lowest: ₪4,100 @ ivory
    - ₪4,200  iPhone 14 Pro 256GB
    ...

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **No API keys** | Fully reproducible, free to run, no vendor lock-in |
| **Character n-gram TF-IDF** | Handles typos, mixed languages, partial names; better than word-level |
| **Category blocking** | Phones cannot be TVs - skip cross-category comparisons |
| **Storage/size rules** | Most common source of false positives; rule-based is reliable |
| **Union-Find** | Handles transitive chains correctly (A=B, B=C → same group) |
| **Conservative threshold** | When in doubt, do not merge (avoids false positives shown to customers) |

---

## Extending

1. **Scale**: Process in chunks; use approximate nearest neighbor (FAISS) instead of dense cosine similarity for 100k+ products  
2. **Accuracy**: Use a fine-tuned sentence-transformer for embedding similarity (`paraphrase-multilingual-MiniLM`)  
3. **Real-time**: Wrap `deduplicate()` in a FastAPI endpoint; cache TF-IDF matrix between requests  

---

*Built with Python · pandas · scikit-learn*  