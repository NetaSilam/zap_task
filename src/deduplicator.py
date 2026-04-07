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
EMBEDDING_THRESHOLD_SAME = 0.70
EMBEDDING_THRESHOLD_CROSS = 0.60

EMBEDDING_MODEL = "paraphrase-multilingual-MiniLM-L12-v2"

# Model-number detection 
_RE_MODEL_NUM = re.compile(r"\b[a-z]*\d+[a-z]*\b|\b\d{2,4}\b")


def _expand_model_tokens(tokens: set[str]) -> set[str]:
    """
    Expand model tokens:
      iphone14pro → 14pro, 14
      q80b        → 80, q80b
    """
    expanded = set(tokens)

    for tok in tokens:
        # extract digit+letters
        m = re.search(r"(\d+[a-z]+)", tok)
        if m:
            expanded.add(m.group(1))

        # extract digits only 
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

    def __post_init__(self):
        self.normalized = normalize(self.name)
        self.storage = extract_storage(self.normalized)
        self.screen = extract_screen_size(self.normalized)

        raw_models = set(_RE_MODEL_NUM.findall(self.normalized))
        self.model_nums = frozenset(_expand_model_tokens(raw_models))


# Union-Find 

class UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

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
        g = {}
        for i in range(len(self.parent)):
            g.setdefault(self.find(i), []).append(i)
        return g


# Helpers 

_RE_HEBREW = re.compile(r"[א-ת]")

def _is_mixed_script(a: Product, b: Product) -> bool:
    return bool(_RE_HEBREW.search(a.normalized)) != bool(_RE_HEBREW.search(b.normalized))


def _shared_model_numbers(a: Product, b: Product) -> bool:
    return bool(a.model_nums & b.model_nums)


def _shared_numbers(a: Product, b: Product) -> bool:
    """
    Strong heuristic:
    If products share numeric tokens,
    it's very likely same model (within same category).
    """
    nums_a = set(re.findall(r"\d+", a.normalized))
    nums_b = set(re.findall(r"\d+", b.normalized))
    return bool(nums_a & nums_b)


# Candidate generation 

def candidate_pairs(products: list[Product]):
    corpus = [p.normalized for p in products]

    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    mat = vec.fit_transform(corpus)
    sim = cosine_similarity(mat)

    pairs = []
    n = len(products)

    for i in range(n):
        for j in range(i + 1, n):
            if products[i].category != products[j].category:
                continue

            nums_i = set(re.findall(r"\d+", products[i].name))
            nums_j = set(re.findall(r"\d+", products[j].name))

            if nums_i & nums_j:
                pairs.append((i, j))
                continue

            if _is_mixed_script(products[i], products[j]):
                pairs.append((i, j))
            elif sim[i, j] >= TFIDF_THRESHOLD:
                pairs.append((i, j))

    return pairs


# Embeddings 

def load_embedding_model():
    print(f"[*] Loading embedding model: {EMBEDDING_MODEL}")
    return SentenceTransformer(EMBEDDING_MODEL)


def embed_all(model, products):
    return model.encode(
        [p.normalized for p in products],
        normalize_embeddings=True,
        show_progress_bar=False
    )


# Duplicate decision 

def are_duplicates(a: Product, b: Product, emb_a, emb_b) -> bool:

    # Hard reject
    if a.storage and b.storage and a.storage != b.storage:
        return False

    if a.screen and b.screen and a.screen != b.screen:
        return False

    # Strong rules
    if _shared_model_numbers(a, b):
        return True

    if _shared_numbers(a, b):
        return True

    # Embedding similarity
    score = float(np.dot(emb_a, emb_b))

    threshold = (
        EMBEDDING_THRESHOLD_CROSS
        if _is_mixed_script(a, b)
        else EMBEDDING_THRESHOLD_SAME
    )

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

    print(f"[1/5] Loaded {len(products)} products.")

    pairs = candidate_pairs(products)
    print(f"[2/5] {len(pairs)} candidate pairs.")

    model = load_embedding_model()
    embeddings = embed_all(model, products)

    uf = UnionFind(len(products))

    confirmed = 0
    for i, j in pairs:
        if are_duplicates(products[i], products[j], embeddings[i], embeddings[j]):
            uf.union(i, j)
            confirmed += 1

    print(f"[3/5] {confirmed} duplicate pairs confirmed.")

    rows = []
    for gid, members in enumerate(uf.groups().values()):
        group = [products[m] for m in members]
        cheapest = min(group, key=lambda p: p.price)

        rows.append({
            "group_id": gid,
            "canonical_name": cheapest.name,
            "lowest_price": cheapest.price,
            "cheapest_store": cheapest.store,
            "member_count": len(group),
            "member_ids": [p.id for p in group],
            "member_names": [p.name for p in group],
            "all_prices": [p.price for p in group],
        })

    result = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)

    print(f"[4/5] Done: {len(products)} → {len(result)} groups.")

    return result


# CLI 

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input")
    parser.add_argument("--output", default="output_deduplicated.csv")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    result = deduplicate(df)

    result.to_csv(args.output, index=False, encoding="utf-8-sig")

    print("GROUPS:\n")

    for _, row in result.iterrows():
        print(f"  {row['canonical_name']}  ({row['member_count']} items)")
        print(f"     Lowest: ₪{row['lowest_price']:,.0f} @ {row['cheapest_store']}")

        for name, price in zip(row["member_names"], row["all_prices"]):
            print(f"    - ₪{price:,.0f}  {name}")

        print()