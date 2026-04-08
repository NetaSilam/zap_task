# Zap Product Deduplication Engine

> **GenAI Exploration Lead - Home Assignment**

---

## Approach

### 1. The Problem
A product list contains duplicates with inconsistent names (Hebrew/English mixed, different word order, extra descriptors).  
The goal is to merge duplicates and surface the lowest available price.

### 2. Solution - 5-Step Pipeline

1. **Normalization (`utils.py`)**
   - Lowercase, strip diacritics, remove punctuation
   - Split concatenated tokens (`iphone14pro` → `iphone 14 pro`)
   - POS-based noise removal via spaCy (English) + static list (Hebrew)
   - No brand translation - the multilingual embedding model handles Hebrew/English natively

2. **Candidate Blocking (`deduplicator.py`)**
   - Priority-ordered rules to avoid O(n²) full comparison:
     1. Shared 3+ digit numbers (e.g. `1000` in `WH-1000XM5`)
     2. Shared model-ID tokens (e.g. `v15`, `s23`, `q80b`)
     3. Mixed-script pairs (Hebrew ↔ English) - always forwarded to verification
     4. TF-IDF character n-gram cosine similarity fallback

3. **Multilingual Embedding Verification**
   - Model: `paraphrase-multilingual-MiniLM-L12-v2` (50+ languages, ~120 MB, cached locally)
   - Adaptive thresholds by pair type:
     - Mixed-script (Hebrew ↔ English): `0.60`
     - Same-script with shared model identifier: `0.70`
     - Same-script general: `0.82`

4. **Hard Rules**
   - Storage mismatch → reject (`128GB ≠ 256GB`)
   - Screen size mismatch → reject (`55" ≠ 65"`)
   - Mixed-script + shared model token → fast-accept (no embedding needed)

5. **Union-Find Grouping + Aggregation**
   - Merges confirmed duplicate pairs
   - Handles transitive chains: A=B, B=C → A=B=C
   - Surfaces the lowest price and cheapest store per group

---

## Project Structure

```
zap-dedup/
├── data/
│   └── products_sample.csv   # 20 sample products with deliberate duplicates
├── src/
│   ├── deduplicator.py       # Main pipeline
│   └── utils.py              # Normalization helpers
├── tests/
│   └── test_deduplicator.py  # pytest - covers normalize, rules, full pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
python -m spacy download en_core_web_sm   # optional - improves English noise removal
```

### 2. Run on sample data
```bash
cd src
python deduplicator.py ../data/products_sample.csv --output results.csv
```

### 3. Run tests
```bash
pytest ../tests/
```

### Example Output

```
[1/5] Loaded 20 products.
[2/5] 40 candidate pairs.
[3/5] 19 duplicate pairs confirmed.
[4/5] Done: 20 listings -> 7 unique products (6 groups had duplicates).

GROUPS:

  Samsung S23 256GB  (4 items)
     Lowest: 2,850 @ ivory
    -> 2,850  Samsung S23 256GB
       2,999  Samsung Galaxy S23
       3,100  סמסונג גלקסי S23
       3,200  galaxy s23 samsung

  Apple iPhone 14 Pro  (4 items)
     Lowest: 4,100 @ ivory
       4,200  iPhone 14 Pro 256GB
       4,500  אייפון 14 פרו
    -> 4,100  Apple iPhone 14 Pro
       4,300  iphone14pro 256
  ...
```

---

## Design Decisions

| Decision | Rationale |
|---|---|
| **No API keys** | Fully reproducible, free to run, no vendor lock-in |
| **spaCy POS tagging** | Detects noise words dynamically (ADJ/ADV) without a manual dictionary |
| **Adaptive thresholds** | Mixed-script pairs need a lower bar - the embedding model is the only cross-language signal |
| **Fast-accept rule** | Mixed-script + shared model token (e.g. `דייסון`/`Dyson` + `v15`) → accept without embedding |
| **Character n-gram TF-IDF** | Handles typos, partial names; better than word-level for Hebrew/English |
| **Category blocking** | Phones cannot be TVs - skips cross-category comparisons |
| **Storage/size hard rules** | Most common source of false positives; rule-based is more reliable than embeddings here |
| **Union-Find** | Handles transitive chains correctly (A=B, B=C → same group) |
| **Conservative threshold** | When in doubt, do not merge - false positives shown to customers are worse than missed merges |

---

## Extending

1. **Scale** - Use FAISS approximate nearest neighbor instead of dense cosine similarity for 100k+ products
2. **Hebrew NLP** - Add `he_core_news_sm` spaCy model for POS-based Hebrew noise removal (currently uses static list)
3. **Real-time** - Wrap `deduplicate()` in a FastAPI endpoint; cache TF-IDF matrix between requests
4. **Confidence scores** - Expose the embedding similarity score per group for human review of borderline cases

---

*Built with Python · pandas · scikit-learn · sentence-transformers · spaCy*
