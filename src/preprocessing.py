# src/preprocessing.py

from __future__ import annotations

import re
import unicodedata
from typing import Iterable

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

# Weak business/legal-form tokens.
#
# IMPORTANT:
# These are NOT deleted from the canonical normalized text.
# They are only excluded from the name-token blocking signal.
BLOCK_STOP_TOKENS = frozenset({
    "inc",
    "incorporated",
    "llc",
    "ltd",
    "limited",
    "private",
    "pvt",
    "corp",
    "corporation",
    "company",
    "co",
    "plc",
    "llp",
    "trust",
    "group",
    "holdings",
    "services",
})


# ============================================================
# BASIC TEXT NORMALIZATION
# ============================================================

def normalize_text(value) -> str | pd.NA:
    """
    Canonical normalization for a single text value.

    Steps:
        1. Handle missing values.
        2. Unicode NFKC normalization.
        3. Unicode casefold.
        4. Replace non-letter/non-mark/non-number characters
           with spaces.
        5. Collapse repeated whitespace.
        6. Strip leading/trailing whitespace.

    Returns:
        normalized string or pd.NA
    """

    if pd.isna(value):
        return pd.NA

    value = str(value)

    # Unicode compatibility normalization
    value = unicodedata.normalize("NFKC", value)

    # Case-insensitive normalization
    value = value.casefold()

    # Preserve:
    #   L = Letter
    #   M = Mark
    #   N = Number
    #
    # Replace everything else with whitespace.
    chars = []

    for char in value:
        category = unicodedata.category(char)

        if category[0] in ("L", "M", "N"):
            chars.append(char)
        else:
            chars.append(" ")

    value = "".join(chars)

    # Collapse whitespace
    value = re.sub(r"\s+", " ", value)

    return value.strip()


def normalize_text_series(series: pd.Series) -> pd.Series:
    """
    Apply canonical text normalization to a pandas Series.
    """

    return series.astype("string").map(
        normalize_text,
        na_action="ignore",
    ).astype("string")


# ============================================================
# NAME REPRESENTATIONS
# ============================================================

def name_tokens(
    value,
    remove_stop_tokens: bool = True,
    min_token_length: int = 3,
) -> list[str]:
    """
    Create canonical name tokens.

    Stop tokens are removed only from this blocking representation.
    The original normalized name remains unchanged.
    """

    normalized = normalize_text(value)

    if pd.isna(normalized) or not normalized:
        return []

    tokens = normalized.split()

    if remove_stop_tokens:
        tokens = [
            token
            for token in tokens
            if token not in BLOCK_STOP_TOKENS
        ]

    tokens = [
        token
        for token in tokens
        if len(token) >= min_token_length
    ]

    # Deterministic ordering and duplicate removal.
    return sorted(set(tokens))


def char_ngrams(
    value,
    n: int = 3,
) -> list[str]:
    """
    Create character n-grams from canonical normalized text.

    Spaces are retained because they contain useful boundary
    information between words.
    """

    normalized = normalize_text(value)

    if pd.isna(normalized) or len(normalized) < n:
        return []

    return sorted({
        normalized[i:i + n]
        for i in range(len(normalized) - n + 1)
    })


# ============================================================
# ADDRESS REPRESENTATION
# ============================================================

def address_tokens(
    value,
    min_token_length: int = 2,
) -> list[str]:
    """
    Token representation for addresses.

    Unlike business-name blocking, legal business suffixes are
    not removed here.
    """

    normalized = normalize_text(value)

    if pd.isna(normalized) or not normalized:
        return []

    tokens = normalized.split()

    tokens = [
        token
        for token in tokens
        if len(token) >= min_token_length
    ]

    return sorted(set(tokens))


# ============================================================
# COUNTRY
# ============================================================

def normalize_country(value) -> str | pd.NA:
    """
    Normalize country as a categorical/string field.

    We intentionally do not restrict this to US/India.
    Test data contains France, so country handling must remain
    open-set.
    """

    normalized = normalize_text(value)

    if pd.isna(normalized) or not normalized:
        return pd.NA

    return normalized


# ============================================================
# DATAFRAME ENRICHMENT
# ============================================================

def add_name_features(
    df: pd.DataFrame,
    name_column: str = "business_name",
) -> pd.DataFrame:
    """
    Add canonical name representations.
    """

    result = df.copy()

    result["name_norm"] = normalize_text_series(
        result[name_column]
    )

    result["name_tokens"] = result["name_norm"].map(
        lambda x: name_tokens(x)
        if not pd.isna(x)
        else []
    )

    result["char_ngrams"] = result["name_norm"].map(
        lambda x: char_ngrams(x, n=3)
        if not pd.isna(x)
        else []
    )

    return result


def add_address_features(
    df: pd.DataFrame,
    address_column: str = "business_address",
) -> pd.DataFrame:
    """
    Add canonical address representations.
    """

    result = df.copy()

    result["address_norm"] = normalize_text_series(
        result[address_column]
    )

    result["address_tokens"] = result["address_norm"].map(
        lambda x: address_tokens(x)
        if not pd.isna(x)
        else []
    )

    return result


def add_country_features(
    df: pd.DataFrame,
    country_column: str = "country",
) -> pd.DataFrame:
    """
    Add canonical country representation.
    """

    result = df.copy()

    result["country_norm"] = result[country_column].map(
        normalize_country
    )

    return result


def enrich_entity_dataframe(
    df: pd.DataFrame,
    name_column: str = "business_name",
    address_column: str = "business_address",
    country_column: str = "country",
) -> pd.DataFrame:
    """
    Apply the complete canonical preprocessing pipeline.

    This is the main function that should be used by the rest
    of the project.
    """

    result = df.copy()

    result = add_name_features(
        result,
        name_column=name_column,
    )

    result = add_address_features(
        result,
        address_column=address_column,
    )

    result = add_country_features(
        result,
        country_column=country_column,
    )

    return result