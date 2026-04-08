"""
Unit tests - run with: pytest tests/
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from utils import normalize, extract_storage, extract_screen_size
from deduplicator import (
    Product, are_duplicates, candidate_pairs, deduplicate, load_embedding_model
)


# Shared embedding model 

@pytest.fixture(scope="session")
def model():
    return load_embedding_model()


def embed(model, *names):
    return model.encode(list(names), normalize_embeddings=True)


# normalize()

class TestNormalize:
    def test_lowercase(self):
        assert normalize("Samsung") == "samsung"

    def test_hebrew_brand(self):
        result = normalize("סמסונג גלקסי S23")
        assert "סמסונג" in result or "s23" in result

    def test_strips_noise(self):
        r = normalize("iPhone 14 חדש אחריות")
        assert "חדש" not in r and "אחריות" not in r

    def test_collapses_whitespace(self):
        assert "  " not in normalize("sony   wh  1000")

    def test_removes_punctuation(self):
        assert "!" not in normalize("Samsung Galaxy S23!")


# extract helpers

class TestExtractStorage:
    def test_gb(self): assert extract_storage("iPhone 256GB") == "256gb"
    def test_tb(self): assert extract_storage("SSD 1TB") == "1024gb"
    def test_none(self): assert extract_storage("LG TV 55 inch") is None

class TestExtractScreenSize:
    def test_inch(self): assert extract_screen_size("LG 55 inch") == "55"
    def test_quote(self): assert extract_screen_size('Samsung 65"') == "65"
    def test_none(self): assert extract_screen_size("Sony WH-1000XM5") is None


# are_duplicates()

def make(name, price=1000, cat="phones", store="KSP", pid=1):
    return Product(id=pid, name=name, price=price, category=cat, store=store)

class TestAreDuplicates:
    def test_same_product_hebrew_english(self, model):
        a = make("Samsung Galaxy S23")
        b = make("סמסונג גלקסי S23")
        embs = embed(model, a.name, b.name)
        assert are_duplicates(a, b, embs[0], embs[1]) is True

    def test_different_storage(self, model):
        a = make("iPhone 14 Pro 128GB")
        b = make("iPhone 14 Pro 256GB")
        embs = embed(model, a.name, b.name)
        assert are_duplicates(a, b, embs[0], embs[1]) is False

    def test_different_screen(self, model):
        a = make("LG OLED 55 inch", cat="tv")
        b = make("LG OLED 65 inch", cat="tv")
        embs = embed(model, a.name, b.name)
        assert are_duplicates(a, b, embs[0], embs[1]) is False

    def test_completely_different(self, model):
        a = make("Dyson V15 Detect", cat="vacuum")
        b = make("Sony WH-1000XM5", cat="audio")
        embs = embed(model, a.name, b.name)
        assert are_duplicates(a, b, embs[0], embs[1]) is False


# full pipeline

class TestDeduplicate:
    def test_sample_data(self):
        df = pd.read_csv(
            Path(__file__).parent.parent / "data" / "products_sample.csv"
        )
        result = deduplicate(df)
        assert len(result) < len(df)
        assert result["canonical_name"].notna().all()
        for _, row in result.iterrows():
            assert row["lowest_price"] == min(row["all_prices"])
