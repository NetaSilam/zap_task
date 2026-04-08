"""
Deduplicate product listings and surface lowest price.

Algorithm Overview:
    1. Normalize
    2. TF-IDF blocking
    3. Embedding verification
    4. Hard rules:
       - Reject: storage / screen mismatch
       - Accept: shared model numbers (expanded)
       - Accept: shared numeric tokens (strong heuristic)
    5. Union-Find grouping
    6. Aggregate results
"""

from __future__ import annotations

import argparse
import re
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import normalize, extract_storage, extract_screen_size

# Thresholds

TFIDF_THRESHOLD = 0.10
EMBEDDING_THRESHOLD_SAME = 0.82   
EMBEDDING_THRESHOLD_CROSS = 0.60 
EMBEDDING_THRESHOLD_SOFT = 0.70   

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Model-number patterns
_RE_MODEL_NUM = re.compile(r"\b[a-z]*\d+[a-z]*\b|\b\d{2,4}\b")
# Letter+digit or digit+letter combos 
_RE_MODEL_ID = re.compile(r"[a-z]+\d+|\d+[a-z]+", re.IGNORECASE)

_STORAGE_VALS = frozenset({
    "128", "256", "512", "1024", "2048",
    "128gb", "256gb", "512gb", "1024gb",
})


def _expand_model_tokens(tokens: set[str]) -> set[str]:
    expanded = set(tokens)
    for tok in tokens:
        m = re.search(r"(\d+[a-z]+)", tok)
        if m:
            expanded.add(m.group(1))
        m2 = re.search(r"(\d+)", tok)
        if m2:
            expanded.add(m2.group(1))
    return expanded


# Product 

@dataclass
class Product:
    id: int
    name: str
    price: float
    category: str
    store: str
    normalized: str = field(init=False)
    storage: str | None = field(init=False)
    screen: str | None = field(init=False)
    model_nums: frozenset[str] = field(init=False)
    # Pre-computed model-ID tokens from the *normalized* name (lower-case)
    model_ids: frozenset[str] = field(init=False)
    # Pre-computed 3+-digit numbers (excluding storage) from the raw name
    long_nums: frozenset[str] = field(init=False)

    def __post_init__(self):
        self.normalized = normalize(self.name)
        self.storage = extract_storage(self.normalized)
        self.screen = extract_screen_size(self.normalized)

        raw_models = set(_RE_MODEL_NUM.findall(self.normalized))
        self.model_nums = frozenset(_expand_model_tokens(raw_models))

        self.model_ids = frozenset(
            m.lower() for m in _RE_MODEL_ID.findall(self.normalized)
        )
        self.long_nums = frozenset(
            n for n in re.findall(r"\d+", self.name) if len(n) >= 3
        ) - _STORAGE_VALS


# Union-Find 

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank   = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int):
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self):
        g: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            g.setdefault(self.find(i), []).append(i)
        return g


# Helpers 

_RE_HEBREW = re.compile(r"[א-ת]")


def _is_hebrew(text: str) -> bool:
    return bool(_RE_HEBREW.search(text))


def _is_mixed_script(a: Product, b: Product) -> bool:
    return _is_hebrew(a.normalized) != _is_hebrew(b.normalized)


def _shared_model_numbers(a: Product, b: Product) -> bool:
    return bool((a.model_nums - _STORAGE_VALS) & (b.model_nums - _STORAGE_VALS))


# Candidate generation 

def candidate_pairs(products: list[Product]) -> list[tuple[int, int]]:
    """
    Build the set of (i, j) pairs to send to embedding verification.

    Priority order (first match wins, avoids duplicates in list):
      1. Shared 3+ digit non-storage numbers 
      2. Shared model-ID tokens               
      3. Mixed-script pair in same category   
      4. TF-IDF similarity >= threshold       
    """
    corpus = [p.normalized for p in products]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    mat = vec.fit_transform(corpus)
    sim = cosine_similarity(mat)

    pairs: list[tuple[int, int]] = []
    n = len(products)

    for i in range(n):
        for j in range(i + 1, n):
            if products[i].category != products[j].category:
                continue

            pi, pj = products[i], products[j]

            # 1. Shared 3+ digit numbers 
            if pi.long_nums & pj.long_nums:
                pairs.append((i, j))
                continue

            # 2. Shared model-ID tokens 
            if pi.model_ids & pj.model_ids:
                pairs.append((i, j))
                continue

            # 3. Mixed-script - always send to embedding
            if _is_mixed_script(pi, pj):
                pairs.append((i, j))
                continue

            # 4. TF-IDF fallback for same-script pairs
            if sim[i, j] >= TFIDF_THRESHOLD:
                pairs.append((i, j))

    return pairs


# Embeddings 

def load_embedding_model() -> SentenceTransformer:
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_all(model: SentenceTransformer, products: list[Product]) -> np.ndarray:
    return model.encode(
        [p.normalized for p in products],
        normalize_embeddings=True,
        show_progress_bar=False,
    )


# Duplicate decision 

def are_duplicates(a: Product, b: Product, emb_a: np.ndarray, emb_b: np.ndarray) -> bool:
    # Hard rejects 
    if a.storage and b.storage and a.storage != b.storage:
        return False
    if a.screen and b.screen and a.screen != b.screen:
        return False

    # Fast-accept: mixed-script + shared model-ID token 
    if _is_mixed_script(a, b) and (a.model_ids & b.model_ids):
        return True

    score = float(np.dot(emb_a, emb_b))

    if _is_mixed_script(a, b):
        threshold = EMBEDDING_THRESHOLD_CROSS        
    elif _shared_model_numbers(a, b):
        threshold = EMBEDDING_THRESHOLD_SOFT        
    else:
        nums_a = set(re.findall(r"\d+", a.normalized)) - _STORAGE_VALS
        nums_b = set(re.findall(r"\d+", b.normalized)) - _STORAGE_VALS
        threshold = EMBEDDING_THRESHOLD_SOFT if (nums_a & nums_b) else EMBEDDING_THRESHOLD_SAME

    return score >= threshold


# Main pipeline 

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    products = [
        Product(
            id=int(row["id"]),
            name=str(row["name"]),
            price=float(row["price"]),
            category=str(row["category"]),
            store=str(row["store"]),
        )
        for _, row in df.iterrows()
    ]
    print(f"[1/4] Loaded {len(products)} products.")

    pairs = candidate_pairs(products)
    print(f"[2/4] {len(pairs)} candidate pairs.")

    model = load_embedding_model()
    embeddings = embed_all(model, products)

    uf = UnionFind(len(products))
    confirmed = 0
    for i, j in pairs:
        if are_duplicates(products[i], products[j], embeddings[i], embeddings[j]):
            uf.union(i, j)
            confirmed += 1
    print(f"[3/4] {confirmed} duplicate pairs confirmed.")

    rows = []
    for gid, members in enumerate(uf.groups().values()):
        group    = [products[m] for m in members]
        cheapest = min(group, key=lambda p: p.price)
        rows.append({
            "group_id": gid,
            "canonical_name": cheapest.name,
            "lowest_price": cheapest.price,
            "cheapest_store": cheapest.store,
            "member_count": len(group),
            "member_ids": [p.id    for p in group],
            "member_names": [p.name  for p in group],
            "all_prices": [p.price for p in group],
        })

    result = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)
    dupes  = result[result["member_count"] > 1]
    print(f"[4/4] Done: {len(products)} listings -> {len(result)} unique products "
          f"({len(dupes)} groups had duplicates).")
    return result


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="output_deduplicated.csv")
    args = parser.parse_args()

    df     = pd.read_csv(args.input)
    result = deduplicate(df)
    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("\nGROUPS:\n")
    for _, row in result.iterrows():
        if row["member_count"] > 1:
            print(f"  {row['canonical_name']}  ({row['member_count']} items)")
            print(f"     Lowest: {row['lowest_price']:,.0f} @ {row['cheapest_store']}")
            for name, price in zip(row["member_names"], row["all_prices"]):
                marker = "->" if price == row["lowest_price"] else "  "
                print(f"    {marker} {price:,.0f}  {name}")
            print()