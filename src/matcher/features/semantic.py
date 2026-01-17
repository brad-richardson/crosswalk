"""Semantic feature extraction (names, classifications)."""

import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# Road class mapping to hierarchy levels
ROAD_CLASS_HIERARCHY = {
    "motorway": 1,
    "trunk": 2,
    "primary": 3,
    "secondary": 4,
    "tertiary": 5,
    "residential": 6,
    "service": 7,
    "unclassified": 8,
    "track": 9,
    "path": 10,
}

# Common street name abbreviations
# Note: Keys must include trailing space to avoid matching inside words
# (e.g., " st " won't match inside "street")
STREET_ABBREVIATIONS = {
    " st ": " street ",
    " st. ": " street ",
    " rd ": " road ",
    " rd. ": " road ",
    " ave ": " avenue ",
    " ave. ": " avenue ",
    " blvd ": " boulevard ",
    " blvd. ": " boulevard ",
    " dr ": " drive ",
    " dr. ": " drive ",
    " ln ": " lane ",
    " ln. ": " lane ",
    " ct ": " court ",
    " ct. ": " court ",
    " pl ": " place ",
    " pl. ": " place ",
    " cir ": " circle ",
    " cir. ": " circle ",
    " hwy ": " highway ",
    " hwy. ": " highway ",
    " pkwy ": " parkway ",
    " pkwy. ": " parkway ",
    " n ": " north ",
    " n. ": " north ",
    " s ": " south ",
    " s. ": " south ",
    " e ": " east ",
    " e. ": " east ",
    " w ": " west ",
    " w. ": " west ",
    " ne ": " northeast ",
    " nw ": " northwest ",
    " se ": " southeast ",
    " sw ": " southwest ",
}


# Default result when names are missing - use neutral scores (0.5)
# to avoid penalizing valid geometric matches when one dataset lacks
# name data. The 'names_missing' flag allows the ML model to handle
# this case specifically if needed.
_MISSING_NAMES_RESULT = {
    "levenshtein_ratio": 0.5,
    "jaro_winkler": 0.5,
    "token_sort_ratio": 0.5,
    "token_set_ratio": 0.5,
    "partial_ratio": 0.5,
    "soundex_match": 0.5,
    "metaphone_similarity": 0.5,
    "names_match": False,
    "names_missing": True,
}


def _extract_name_string(name) -> str | None:
    """Extract string from name, handling dict format.

    Overture/OSM data often stores names as dicts like:
    - {'primary': 'Main Street'}
    - {'primary': 'Main St', 'common': None, 'rules': [...]}

    Args:
        name: String or dict containing name

    Returns:
        Extracted name string or None
    """
    if name is None:
        return None
    if isinstance(name, str):
        return name
    if isinstance(name, dict):
        # Try common keys in order of preference
        for key in ["primary", "common", "name", "value"]:
            if key in name and name[key]:
                val = name[key]
                # Handle nested extraction
                if isinstance(val, str):
                    return val
                if isinstance(val, dict):
                    return _extract_name_string(val)
        # Last resort - return first non-None string value
        for v in name.values():
            if isinstance(v, str) and v:
                return v
    return None


def compute_name_similarity(
    name_a,
    name_b,
) -> dict[str, float]:
    """Compute multiple string similarity metrics.

    Args:
        name_a: First street name (string or dict with 'primary' key)
        name_b: Second street name (string or dict with 'primary' key)

    Returns:
        Dictionary with:
            - levenshtein_ratio: Normalized Levenshtein distance (0-1)
            - jaro_winkler: Jaro-Winkler similarity (0-1)
            - token_sort_ratio: Token-sorted fuzzy ratio (0-1)
            - token_set_ratio: Token set ratio (0-1, handles subsets)
            - partial_ratio: Partial string ratio (0-1)
    """
    # Extract string from dict if needed
    name_a = _extract_name_string(name_a)
    name_b = _extract_name_string(name_b)

    if not name_a or not name_b:
        # Return neutral scores when names are missing
        # This prevents penalizing valid geometric matches just because
        # one dataset doesn't have name data for this segment
        return _MISSING_NAMES_RESULT.copy()

    # Normalize names
    norm_a = _normalize_street_name(name_a)
    norm_b = _normalize_street_name(name_b)

    # Handle empty after normalization (e.g., name was just punctuation)
    if not norm_a or not norm_b:
        return _MISSING_NAMES_RESULT.copy()

    # Compute various similarity metrics
    levenshtein_ratio = fuzz.ratio(norm_a, norm_b) / 100.0
    jaro_winkler = JaroWinkler.normalized_similarity(norm_a, norm_b)
    token_sort_ratio = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
    token_set_ratio = fuzz.token_set_ratio(norm_a, norm_b) / 100.0
    partial_ratio = fuzz.partial_ratio(norm_a, norm_b) / 100.0

    # Phonetic matching - catches typos and transcription errors
    # Use first word for Soundex (usually the main street name)
    first_word_a = norm_a.split()[0] if norm_a else ""
    first_word_b = norm_b.split()[0] if norm_b else ""
    soundex_a = jellyfish.soundex(first_word_a) if first_word_a else ""
    soundex_b = jellyfish.soundex(first_word_b) if first_word_b else ""
    soundex_match = 1.0 if soundex_a == soundex_b and soundex_a else 0.0

    # Metaphone on full name for better typo tolerance
    metaphone_a = jellyfish.metaphone(norm_a) if norm_a else ""
    metaphone_b = jellyfish.metaphone(norm_b) if norm_b else ""
    metaphone_similarity = (
        fuzz.ratio(metaphone_a, metaphone_b) / 100.0 if metaphone_a and metaphone_b else 0.5
    )

    # Names match if any metric is very high
    names_match = levenshtein_ratio > 0.9 or token_sort_ratio > 0.9 or token_set_ratio > 0.95

    return {
        "levenshtein_ratio": levenshtein_ratio,
        "jaro_winkler": jaro_winkler,
        "token_sort_ratio": token_sort_ratio,
        "token_set_ratio": token_set_ratio,
        "partial_ratio": partial_ratio,
        "soundex_match": soundex_match,
        "metaphone_similarity": metaphone_similarity,
        "names_match": names_match,
        "names_missing": False,
    }


def _normalize_street_name(name: str) -> str:
    """Normalize street name for comparison.

    - Convert to lowercase
    - Expand abbreviations
    - Remove extra whitespace
    - Remove common punctuation
    """
    if not name:
        return ""

    # Lowercase
    name = name.lower().strip()

    # Remove common punctuation
    name = name.replace(".", "").replace(",", "").replace("-", " ")

    # Add spaces around name for abbreviation matching
    name = f" {name} "

    # Expand abbreviations
    for abbr, full in STREET_ABBREVIATIONS.items():
        name = name.replace(abbr, full)

    # Clean up extra whitespace
    name = " ".join(name.split())

    return name


def compute_class_similarity(
    class_a: str | None,
    class_b: str | None,
    subclass_a: str | None = None,
    subclass_b: str | None = None,
) -> float:
    """Compute road class similarity (0-1).

    Higher similarity for same or adjacent classes in the hierarchy.
    When subclasses are provided, they affect the score for same-class pairs.

    Args:
        class_a: First road class
        class_b: Second road class
        subclass_a: First subclass (e.g., sidewalk, crosswalk)
        subclass_b: Second subclass

    Returns:
        Similarity score (0-1)
    """
    if not class_a or not class_b:
        return 0.5  # Unknown, neutral

    class_a = class_a.lower().strip()
    class_b = class_b.lower().strip()

    # Treat "unknown" as neutral - don't penalize or reward
    if class_a == "unknown" or class_b == "unknown":
        return 0.5

    # Exact class match - check subclass for finer distinction
    if class_a == class_b:
        # Normalize subclass values (handle None, NaN, non-str)
        sub_a = subclass_a.lower().strip() if isinstance(subclass_a, str) else None
        sub_b = subclass_b.lower().strip() if isinstance(subclass_b, str) else None

        # If neither has subclass, or subclasses match, full score
        if not sub_a and not sub_b:
            return 1.0
        if sub_a and sub_b:
            if sub_a == sub_b:
                return 1.0
            # Same class, different subclass (e.g., sidewalk vs crosswalk)
            return 0.85
        # One has subclass, other doesn't - slight penalty
        return 0.9

    # Get hierarchy ranks
    rank_a = ROAD_CLASS_HIERARCHY.get(class_a, 5)
    rank_b = ROAD_CLASS_HIERARCHY.get(class_b, 5)

    # Compute difference
    diff = abs(rank_a - rank_b)

    # Exponential decay: same = 1.0, 1 level = 0.8, 2 levels = 0.6, etc.
    return max(0.0, 1.0 - diff * 0.2)


def compute_class_match(
    class_a: str | None,
    class_b: str | None,
) -> bool:
    """Check if road classes are compatible for matching.

    More lenient than similarity - allows matching across
    adjacent hierarchy levels.

    Args:
        class_a: First road class
        class_b: Second road class

    Returns:
        True if classes are compatible
    """
    if not class_a or not class_b:
        return True  # Unknown is compatible with anything

    class_a = class_a.lower().strip()
    class_b = class_b.lower().strip()

    if class_a == class_b:
        return True

    # Get hierarchy ranks
    rank_a = ROAD_CLASS_HIERARCHY.get(class_a, 5)
    rank_b = ROAD_CLASS_HIERARCHY.get(class_b, 5)

    # Allow up to 2 levels difference
    return abs(rank_a - rank_b) <= 2


def extract_numeric_suffix(name: str | None) -> int | None:
    """Extract numeric suffix from road name (e.g., 'Interstate 5' -> 5).

    Useful for matching numbered routes.
    """
    if not name:
        return None

    import re

    # Find all numbers in the name
    numbers = re.findall(r"\d+", name)

    if numbers:
        # Return the last number (often the route number)
        return int(numbers[-1])

    return None


def names_likely_same_road(name_a: str | None, name_b: str | None) -> bool:
    """Quick check if two names likely refer to the same road.

    Uses multiple heuristics for a quick yes/no decision.
    """
    if not name_a or not name_b:
        return False

    # Quick exact match
    if name_a.lower().strip() == name_b.lower().strip():
        return True

    # Normalize and compare
    norm_a = _normalize_street_name(name_a)
    norm_b = _normalize_street_name(name_b)

    if norm_a == norm_b:
        return True

    # High token set ratio (handles "Main St" vs "N Main Street")
    if fuzz.token_set_ratio(norm_a, norm_b) >= 90:
        return True

    # Check numeric suffix for numbered routes
    num_a = extract_numeric_suffix(name_a)
    num_b = extract_numeric_suffix(name_b)

    if num_a is not None and num_b is not None and num_a == num_b:
        # Both have same number, check if route-like
        route_words = ["interstate", "highway", "route", "us", "sr", "i-", "hwy"]
        a_is_route = any(w in name_a.lower() for w in route_words)
        b_is_route = any(w in name_b.lower() for w in route_words)

        if a_is_route and b_is_route:
            return True

    return False
