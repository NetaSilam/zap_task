"""
utils.py
========
Text-normalisation helpers — no API keys, no external calls.
"""

from __future__ import annotations
import re
import unicodedata

# Hebrew brand/word → English canonical
HEBREW_TO_ENGLISH: dict[str, str] = {
    "סמסונג": "samsung",
    "אפל": "apple",
    "אייפון": "iphone",
    "גלקסי": "galaxy",
    "שיאומי": "xiaomi",
    "הואוויי": "huawei",
    "אופו": "oppo",
    "גוגל": "google",
    "מוטורולה": "motorola",
    "נוקיה": "nokia",
    "סוני": "sony",
    "בוז": "bose",
    "דייסון": "dyson",
    "בוש": "bosch",
    "פיליפס": "philips",
    "לג": "lg",
    "אלג'י": "lg",
    "פנסוניק": "panasonic",
    "טושיבה": "toshiba",
    "הייסנס": "hisense",
    "טלויזיה": "tv",
    "אוזניות": "headphones",
    "אינץ": "inch",
    "אינץ'": "inch",
}

NOISE_WORDS = {
    "new", "חדש", "best", "price", "מחיר", "משלוח", "חינם",
    "מקורי", "original", "אחריות", "warranty", "מהיר", "fast",
    "אונליין", "online", "smartphone", "סמארטפון",
}


def normalize(text: str) -> str:
    """
    Full normalization pipeline (no API, pure string ops):
      1. Lowercase
      2. Hebrew → English substitutions
      3. Strip diacritics / accents
      4. Remove punctuation (keep alphanumeric + spaces)
      5. Remove noise words
      6. Collapse whitespace
    """
    text = text.lower().strip()

    for heb, eng in HEBREW_TO_ENGLISH.items():
        text = text.replace(heb, eng)

    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # keep letters, digits, Hebrew chars, spaces
    text = re.sub(r"[^a-z0-9א-ת\s]", " ", text)

    tokens = [t for t in text.split() if t not in NOISE_WORDS]
    text = " ".join(tokens)

    return re.sub(r"\s+", " ", text).strip()


def extract_storage(text: str) -> str | None:
    """Return normalised storage spec e.g. '256gb', '1024gb' (1TB)."""
    m = re.search(r"(\d+)\s*(gb|tb|mb)", text.lower())
    if m:
        val, unit = int(m.group(1)), m.group(2)
        return f"{val * 1024 if unit == 'tb' else val}gb"
    return None


def extract_screen_size(text: str) -> str | None:
    """Return screen size digits if found, e.g. '55', '65'."""
    m = re.search(r"\b(\d{2,3})\s*(?:inch|אינץ|\")?(?:\s|$)", text.lower())
    return m.group(1) if m else None


def price_fmt(price: float, currency: str = "₪") -> str:
    return f"{currency}{price:,.0f}"
