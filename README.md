# Zap Product Deduplication Engine

> **GenAI Exploration Lead - Home Assignment**  

---

## Approach

### The Problem
A product list contains duplicates with inconsistent names (Hebrew/English mixed, different word order, extra descriptors).  
The goal: merge duplicates and show the customer the lowest available price.

### Solution - 4-Step Pipeline 

```
Input CSV
    │
    ▼
┌─────────────────────────────────────────┐
│ 1. Normalization  (utils.py)            │
│    • Hebrew brands → English            │
│      "סמסונג" → "samsung"               │ 
│    • Remove punctuation & noise words   │
│    • Collapse whitespace                │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ 2. TF-IDF Blocking  (deduplicator.py)   │
│    • char n-gram cosine similarity      │
│    • Only compares within same category │
│    • Produces small set of candidates   │
│      (avoids O(n²) full comparison)     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ 3. Rule-based Verification              │
│    • Storage mismatch? → NOT duplicate  │
│      "128GB" ≠ "256GB"                  │
│    • Screen size mismatch? → NOT dup    │
│      "55 inch" ≠ "65 inch"              │
│    • Re-check cosine on tight threshold │
└──────────────────┬──────────────────────┘
                   │
                   ▼
┌─────────────────────────────────────────┐
│ 4. Union-Find Grouping + Aggregate      │
│    • Merge confirmed duplicate pairs    │
│    • Handle chains: A=B, B=C → A=B=C    │
│    • Surface lowest price per group     │
└──────────────────┬──────────────────────┘
                   │
                   ▼
             Output CSV
```

---

## Project Structure

```
zap-dedup/
├── data/
│   └── products_sample.csv   # 20 sample products with deliberate duplicates
├── src/
│   ├── deduplicator.py       # Main pipeline
│   └── utils.py              # Normalization, Hebrew→English map, helpers
├── tests/
│   └── test_deduplicator.py  # pytest — covers normalize, rules, full pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start

```bash
# 1. Install dependencies 
pip install -r requirements.txt

# 2. Run on sample data
cd src
python deduplicator.py ../data/products_sample.csv --output results.csv

# 3. Run tests
pytest ../tests/
```

### Example output

```
[1/4] Loaded 20 products.
[2/4] 12 candidate pairs after TF-IDF blocking.
[3/4] Rule-based verification: 9 duplicate pairs confirmed.
[4/4] Done. 20 listings → 6 unique products (6 groups had duplicates).

📦  Samsung S23 256GB
    ✅ Best price: ₪2,850  @ ivory
    → ₪2,850  Samsung S23 256GB
       ₪2,999  Samsung Galaxy S23
       ₪3,100  סמסונג גלקסי S23
       ₪3,200  galaxy s23 samsung

📦  Apple iPhone 14 Pro
    ✅ Best price: ₪4,100  @ ivory
    ...
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **No API keys** | Fully reproducible, free to run, no vendor lock-in |
| **char n-gram TF-IDF** | Handles typos, mixed languages, partial names - better than word-level |
| **Category blocking** | A phone can never be a TV - skip cross-category comparisons |
| **Storage/size rules** | These are the most common false-positive source; rule-based is 100% reliable |
| **Union-Find** | Correctly handles transitive chains (A=B, B=C → same group) |
| **Conservative threshold** | When in doubt → don't merge (avoid false positives shown to customer) |

---

## Extending

- **Scale**: Process in chunks; use approximate nearest-neighbour (FAISS) instead of dense cosine for 100k+ products  
- **Accuracy**: Add a fine-tuned sentence-transformer (e.g. `paraphrase-multilingual-MiniLM`) for embedding similarity  
- **Real-time**: Wrap `deduplicate()` in a FastAPI endpoint; cache TF-IDF matrix between requests  

---

*Built with Python · pandas · scikit-learn*
