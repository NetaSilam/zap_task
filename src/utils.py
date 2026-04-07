"""
Text-normalisation helpers.
Noise words are detected via POS tagging,
"""

from __future__ import annotations
import re
import unicodedata

# POS tags treated as noise 
_NOISE_POS = {"ADJ", "ADV", "INTJ", "PART", "SYM", "DET", "PUNCT"}

# Lazy model loader 

_models: dict[str, object] = {}   # cache: lang -> nlp | False


def _load_model(lang: str) -> object:
    """
    Load spaCy model for `lang` ('en' or 'he') once, cache result.
    Returns nlp object or False if unavailable.
    """
    if lang in _models:
        return _models[lang]

    model_name = {"en": "en_core_web_sm", "he": "he_core_news_sm"}.get(lang)
    if model_name is None:
        _models[lang] = False
        return False

    try:
        import spacy
        nlp = spacy.load(model_name, disable=["parser", "ner"])
        print(f"[*] spaCy model loaded ({model_name}).")
        _models[lang] = nlp
    except Exception as e:
        print(f"[!] spaCy model '{model_name}' unavailable ({e}).")
        _models[lang] = False

    return _models[lang]


# Token helpers 

_RE_HEBREW = re.compile(r"[א-ת]")


def _is_hebrew_token(tok: str) -> bool:
    return bool(_RE_HEBREW.search(tok))


def _pos_tag_segment(text: str, lang: str) -> dict[str, str]:
    """
    Run spaCy POS tagging and return {token_lower: POS}.
    Returns empty dict if model unavailable.
    """
    nlp = _load_model(lang)
    if not nlp or not text.strip():
        return {}
    doc = nlp(text)
    return {tok.text.lower(): tok.pos_ for tok in doc}


def _split_concatenated(text: str) -> str:
    """
    Split concatenated tokens
    """
    text = re.sub(r"([a-zא-ת])(\d)", r"\1 \2", text)
    text = re.sub(r"(\d)([a-zא-ת])", r"\1 \2", text)
    return text


def _normalize_hebrew_brands(text: str) -> str:
    """
    Minimal Hebrew → English brand normalization.
    """
    return (
        text.replace("סמסונג", "samsung")
            .replace("דייסון", "dyson")
            .replace("אייפון", "iphone")
    )


# Public API 

def normalize(text: str) -> str:
    """
    Normalize product name:

      1. Lowercase + strip diacritics
      2. Remove punctuation
      3. Normalize Hebrew brand names (light heuristic)
      4. KEEP both original AND split tokens 
      5. POS-based noise filtering
      6. Collapse whitespace

    Design principle:
    Conservative - prefer false negatives over false positives.
    """
    text = text.lower().strip()

    # Strip accents
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if unicodedata.category(c) != "Mn")

    # Keep alphanumeric + Hebrew
    text = re.sub(r"[^a-z0-9א-ת\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()

    # Hebrew brand normalization
    text = _normalize_hebrew_brands(text)

    # Keep BOTH original and split versions
    split_text = _split_concatenated(text)
    text = f"{text} {split_text}"
    text = re.sub(r"\s+", " ", text).strip()

    # POS tagging by language
    hebrew_segment = " ".join(t for t in text.split() if _is_hebrew_token(t))
    english_segment = " ".join(t for t in text.split() if not _is_hebrew_token(t))

    he_pos = _pos_tag_segment(hebrew_segment, "he")
    en_pos = _pos_tag_segment(english_segment, "en")

    # Filter noise tokens
    tokens = []
    for tok in text.split():
        tok_lower = tok.lower()

        if _is_hebrew_token(tok):
            pos = he_pos.get(tok_lower)
        else:
            pos = en_pos.get(tok_lower)

        # If POS known and considered noise - drop
        if pos is not None and pos in _NOISE_POS:
            continue

        tokens.append(tok)

    return re.sub(r"\s+", " ", " ".join(tokens)).strip()


def extract_storage(text: str) -> str | None:
    """Return storage spec: '256gb', '1024gb'."""
    m = re.search(r"(\d+)\s*(gb|tb|mb)", text.lower())
    if m:
        val, unit = int(m.group(1)), m.group(2)
        return f"{val * 1024 if unit == 'tb' else val}gb"
    return None


def extract_screen_size(text: str) -> str | None:
    """Return screen size: '55', '65'."""
    m = re.search(r"\b(\d{2,3})\s*(?:inch|אינץ|\")?(?:\s|$)", text.lower())
    return m.group(1) if m else None


def price_fmt(price: float, currency: str = "₪") -> str:
    return f"{currency}{price:,.0f}"
