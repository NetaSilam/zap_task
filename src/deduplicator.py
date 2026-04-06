"""
deduplicator.py
===============
Deduplicate product listings and surface the lowest price.

Algorithm — 100% offline, zero API calls:
  1. Normalize  – unify Hebrew/English brand names, strip noise
  2. Block      – TF-IDF char n-gram cosine similarity (candidate pairs)
  3. Verify     – rule-based checks (storage/size mismatch = not duplicate)
  4. Group      – Union-Find merges confirmed duplicates
  5. Aggregate  – lowest price + cheapest store per group
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from utils import normalize, extract_storage, extract_screen_size

# ── Tuning knobs ────────────────────────────────────────────────────────────
SIMILARITY_THRESHOLD = 0.40   # TF-IDF cosine; pairs above go to rule check
# ────────────────────────────────────────────────────────────────────────────


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

    def __post_init__(self):
        self.normalized = normalize(self.name)
        self.storage = extract_storage(self.normalized)
        self.screen = extract_screen_size(self.normalized)


# ── Union-Find ───────────────────────────────────────────────────────────────

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

    def groups(self) -> dict[int, list[int]]:
        g: dict[int, list[int]] = {}
        for i in range(len(self.parent)):
            g.setdefault(self.find(i), []).append(i)
        return g


# ── Blocking: TF-IDF cosine ──────────────────────────────────────────────────

def candidate_pairs(products: list[Product]) -> list[tuple[int, int]]:
    """Return index pairs that are similar enough to inspect further."""
    corpus = [p.normalized for p in products]
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), min_df=1)
    mat = vec.fit_transform(corpus)
    sim = cosine_similarity(mat)

    pairs = []
    n = len(products)
    for i in range(n):
        for j in range(i + 1, n):
            if products[i].category != products[j].category:
                continue   # different categories → never duplicates
            if sim[i, j] >= SIMILARITY_THRESHOLD:
                pairs.append((i, j))
    return pairs


# ── Rule-based verification (no LLM needed) ──────────────────────────────────

def are_duplicates(a: Product, b: Product) -> bool:
    """
    Heuristic rules to confirm or reject a candidate pair.

    Returns True  → same product (merge them)
    Returns False → different products (keep separate)
    """
    # Rule 1: storage mismatch → definitely different SKUs
    if a.storage and b.storage and a.storage != b.storage:
        return False

    # Rule 2: screen size mismatch → different SKUs
    if a.screen and b.screen and a.screen != b.screen:
        return False

    # Rule 3: re-check cosine on normalized text with a tighter threshold
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4))
    try:
        mat = vec.fit_transform([a.normalized, b.normalized])
        score = cosine_similarity(mat[0], mat[1])[0, 0]
    except Exception:
        score = 0.0

    return bool(score >= 0.45)


# ── Main pipeline ─────────────────────────────────────────────────────────────

def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parameters
    ----------
    df : DataFrame with columns [id, name, price, category, store]

    Returns
    -------
    DataFrame with one row per unique product group.
    Columns: group_id, canonical_name, lowest_price, cheapest_store,
             member_count, member_ids, member_names, all_prices
    """
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
    print(f"[2/4] {len(pairs)} candidate pairs after TF-IDF blocking.")

    uf = UnionFind(len(products))
    confirmed = 0
    for i, j in pairs:
        if are_duplicates(products[i], products[j]):
            uf.union(i, j)
            confirmed += 1
    print(f"[3/4] Rule-based verification: {confirmed} duplicate pairs confirmed.")

    rows = []
    for gid, (_, members) in enumerate(uf.groups().items()):
        group = [products[m] for m in members]
        cheapest = min(group, key=lambda p: p.price)
        rows.append({
            "group_id": gid,
            "canonical_name": cheapest.name,   # name from the cheapest listing
            "lowest_price": cheapest.price,
            "cheapest_store": cheapest.store,
            "member_count": len(group),
            "member_ids": [p.id for p in group],
            "member_names": [p.name for p in group],
            "all_prices": [p.price for p in group],
        })

    result = pd.DataFrame(rows).sort_values("group_id").reset_index(drop=True)
    dupes = result[result["member_count"] > 1]
    print(
        f"[4/4] Done. {len(products)} listings → {len(result)} unique products "
        f"({len(dupes)} groups had duplicates)."
    )
    return result


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Deduplicate product listings — no API keys required."
    )
    parser.add_argument("input", help="data/products_sample.csv")
    parser.add_argument(
        "--output", default="output_deduplicated.csv", help="Path to output CSV"
    )
    parser.add_argument(
        "--threshold", type=float, default=SIMILARITY_THRESHOLD,
        help="TF-IDF cosine similarity threshold (default: 0.40)"
    )
    args = parser.parse_args()

    SIMILARITY_THRESHOLD = args.threshold

    df_in = pd.read_csv(args.input)
    df_out = deduplicate(df_in)
    df_out.to_csv(args.output, index=False, encoding="utf-8-sig")

    print(f"\nSaved → {args.output}\n")

    # Pretty-print duplicate groups
    dupes = df_out[df_out["member_count"] > 1]
    for _, row in dupes.iterrows():
        print(f"📦  {row['canonical_name']}")
        print(f"    ✅ Best price: ₪{row['lowest_price']:,.0f}  @ {row['cheapest_store']}")
        for name, price in zip(row["member_names"], row["all_prices"]):
            marker = "→" if price == row["lowest_price"] else "  "
            print(f"    {marker} ₪{price:,.0f}  {name}")
        print()
