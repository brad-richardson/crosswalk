"""Semantic feature extraction (names, classifications)."""

import re
import unicodedata

import jellyfish
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler

# Generic name patterns - roads with these names shouldn't strongly influence matching
# These patterns indicate roads that may share identical names without being the same road
GENERIC_NAME_PATTERNS = [
    r"^unnamed",
    r"^ramp$",
    r"^service\s*(road|rd)?$",
    r"^private\s*(road|rd|drive)?$",
    r"^driveway$",
    r"^alley$",
    r"^connector$",
    r"^access\s*(road|rd)?$",
    r"^frontage\s*(road|rd)?$",
    # Non-vehicular infrastructure
    r"^path$",
    r"^trail$",
    r"^walkway$",
    r"^cycleway$",
    r"^footway$",
    r"^bikeway$",
]

# Pre-compile patterns for efficiency
_GENERIC_NAME_REGEX = re.compile("|".join(GENERIC_NAME_PATTERNS), re.IGNORECASE)

# Road class mapping to hierarchy levels
# Expanded to include link roads, bike/pedestrian infrastructure
# Lower numbers = higher hierarchy (motorways at top)
ROAD_CLASS_HIERARCHY = {
    # Major roads
    "motorway": 1,
    "motorway_link": 1,
    "trunk": 2,
    "trunk_link": 2,
    "primary": 3,
    "primary_link": 3,
    "secondary": 4,
    "secondary_link": 4,
    "tertiary": 5,
    "tertiary_link": 5,
    # Local roads
    "residential": 6,
    "living_street": 6,
    "service": 7,
    "unclassified": 8,
    # Rural/unpaved
    "track": 9,
    # Pedestrian/bike (separate category - use subclass for finer matching)
    "footway": 10,
    "sidewalk": 10,
    "cycleway": 10,
    "path": 10,
    "pedestrian": 10,
    "bridleway": 10,
    "steps": 10,
}

# Traffic tier mapping - groups road classes by traffic type
# This allows for stronger penalties between incompatible traffic types
# (e.g., sidewalks should not match vehicular roads)
TRAFFIC_TIERS = {
    # Vehicular - motorized traffic
    "motorway": "vehicle",
    "motorway_link": "vehicle",
    "trunk": "vehicle",
    "trunk_link": "vehicle",
    "primary": "vehicle",
    "primary_link": "vehicle",
    "secondary": "vehicle",
    "secondary_link": "vehicle",
    "tertiary": "vehicle",
    "tertiary_link": "vehicle",
    "residential": "vehicle",
    "living_street": "vehicle",
    "service": "vehicle",
    "unclassified": "vehicle",
    "track": "vehicle",
    # Bicycle
    "cycleway": "bicycle",
    # Pedestrian
    "footway": "pedestrian",
    "sidewalk": "pedestrian",
    "path": "pedestrian",
    "pedestrian": "pedestrian",
    "steps": "pedestrian",
    # Neutral - uncommon, treat specially
    "bridleway": "neutral",
}

# Cross-tier penalty matrix
# These penalties apply when comparing road classes from different traffic tiers
TIER_PENALTIES = {
    ("vehicle", "pedestrian"): 0.1,  # Strong penalty - incompatible traffic types
    ("pedestrian", "vehicle"): 0.1,
    ("vehicle", "bicycle"): 0.7,  # Mild - bikes often share roads
    ("bicycle", "vehicle"): 0.7,
    ("bicycle", "pedestrian"): 0.5,  # Moderate - shared paths exist
    ("pedestrian", "bicycle"): 0.5,
}


def get_traffic_tier(road_class: str | None) -> str | None:
    """Get traffic tier for a road class.

    Traffic tiers group road classes by traffic type:
    - vehicle: motorized traffic (motorway, residential, etc.)
    - bicycle: dedicated bike infrastructure (cycleway)
    - pedestrian: foot traffic (footway, sidewalk, path, etc.)
    - neutral: uncommon classes treated neutrally (bridleway)

    Args:
        road_class: Road class string (e.g., "residential", "footway")

    Returns:
        Traffic tier string or None if unknown
    """
    if not road_class or not isinstance(road_class, str):
        return None
    return TRAFFIC_TIERS.get(road_class.lower().strip())


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
    # Trail/Terrace/Square
    " tr ": " trail ",
    " trl ": " trail ",
    " ter ": " terrace ",
    " terr ": " terrace ",
    " sq ": " square ",
    # Crossing/Alley
    " xing ": " crossing ",
    " aly ": " alley ",
    # Highway types
    " frwy ": " freeway ",
    " expy ": " expressway ",
    # Mountain/Mount
    " mt ": " mount ",
    " mtn ": " mountain ",
    # Route prefixes
    " sr ": " state route ",
    " cr ": " county road ",
}


# Default result when names are missing - use NaN for similarity scores
# so the ML model (XGBoost) can learn to handle missing names natively.
# has_name_ref/has_name_target encode name presence as binary indicators.
# Road type words to exclude from Soundex — derived from the expanded forms
# already maintained in STREET_ABBREVIATIONS (street, avenue, boulevard, etc.).
_ROAD_TYPE_WORDS = frozenset(
    w for phrase in STREET_ABBREVIATIONS.values() for w in phrase.split()
)


def _soundex_key_word(name: str) -> str:
    """Pick the longest content word for Soundex comparison.

    Filters out road type words (street, avenue, north, etc.) from the expanded
    abbreviations, then returns the longest remaining word. Falls back to the
    longest word overall if all words are road types.
    """
    words = name.split()
    if not words:
        return ""
    content = [w for w in words if w not in _ROAD_TYPE_WORDS]
    return max(content or words, key=len)


_nan = float("nan")
_MISSING_NAMES_RESULT = {
    "levenshtein_ratio": _nan,
    "jaro_winkler": _nan,
    "token_sort_ratio": _nan,
    "token_set_ratio": _nan,
    "partial_ratio": _nan,
    "soundex_match": _nan,
    "metaphone_similarity": _nan,
    "names_match": False,
    "names_missing": True,
    "has_name_ref": 0.0,
    "has_name_target": 0.0,
    "name_is_generic": 0.0,
}


def _is_generic_name(name: str | None) -> bool:
    """Check if a name matches any generic road name pattern.

    Generic names (e.g., 'Unnamed', 'Ramp', 'Service Road') shouldn't
    strongly influence matching because many unrelated roads share them.

    Args:
        name: Street name to check

    Returns:
        True if name matches a generic pattern
    """
    if not name:
        return False
    return bool(_GENERIC_NAME_REGEX.match(name.strip()))


def _get_char_script(ch: str) -> str | None:
    """Get the Unicode script category for an alphabetic character.

    Uses the first word of unicodedata.name(), which identifies the script:
    LATIN, CJK, ARABIC, CYRILLIC, HANGUL, HIRAGANA, KATAKANA, DEVANAGARI, THAI, etc.
    """
    if not ch.isalpha():
        return None
    name = unicodedata.name(ch, "")
    return name.split()[0] if name else None


def _get_text_scripts(text: str) -> set[str]:
    """Get the set of Unicode script categories used in text.

    Returns script names like {"LATIN"}, {"CJK", "HIRAGANA"}, {"ARABIC"}, etc.
    Non-alphabetic characters (digits, punctuation, spaces) are ignored.
    """
    return {s for ch in text if (s := _get_char_script(ch)) is not None}


def _has_non_latin_alpha(text: str) -> bool:
    """Check if text contains non-Latin alphabetic characters."""
    return bool(_get_text_scripts(text) - {"LATIN"})


def _names_are_cross_script(name_a: str, name_b: str) -> bool:
    """Check if two names use different writing systems.

    Extracts Unicode script categories (LATIN, CJK, ARABIC, CYRILLIC, etc.)
    from each name and checks for overlap. No shared scripts means
    character-level similarity metrics will be unreliable.

    Handles mixed-script text correctly: "北京 Beijing Road" shares LATIN
    with "Queen's Road Central", so they are NOT considered cross-script.
    """
    scripts_a = _get_text_scripts(name_a)
    scripts_b = _get_text_scripts(name_b)

    if not scripts_a or not scripts_b:
        return False

    return not scripts_a & scripts_b


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
        name_a: First street name (string or dict with 'primary' key) - reference
        name_b: Second street name (string or dict with 'primary' key) - target

    Returns:
        Dictionary with:
            - levenshtein_ratio: Normalized Levenshtein distance (0-1)
            - jaro_winkler: Jaro-Winkler similarity (0-1)
            - token_sort_ratio: Token-sorted fuzzy ratio (0-1)
            - token_set_ratio: Token set ratio (0-1, handles subsets)
            - partial_ratio: Partial string ratio (0-1)
            - soundex_match: 1.0 if first words have same soundex code
            - metaphone_similarity: Similarity of metaphone encodings (0-1)
            - has_name_ref: 1.0 if reference has non-empty name
            - has_name_target: 1.0 if target has non-empty name
            - name_is_generic: 1.0 if either name matches generic pattern
    """
    # Extract string from dict if needed
    name_a = _extract_name_string(name_a)
    name_b = _extract_name_string(name_b)

    # Compute name presence flags
    has_name_ref = 1.0 if name_a else 0.0
    has_name_target = 1.0 if name_b else 0.0

    if not name_a or not name_b:
        # Return neutral scores when names are missing
        # This prevents penalizing valid geometric matches just because
        # one dataset doesn't have name data for this segment
        result = _MISSING_NAMES_RESULT.copy()
        result["has_name_ref"] = has_name_ref
        result["has_name_target"] = has_name_target
        return result

    # Normalize names
    norm_a = _normalize_street_name(name_a)
    norm_b = _normalize_street_name(name_b)

    # Handle empty after normalization (e.g., name was just punctuation)
    if not norm_a or not norm_b:
        result = _MISSING_NAMES_RESULT.copy()
        result["has_name_ref"] = has_name_ref
        result["has_name_target"] = has_name_target
        return result

    # Check if either name is generic
    name_is_generic = 1.0 if (_is_generic_name(name_a) or _is_generic_name(name_b)) else 0.0

    # Compute various similarity metrics
    levenshtein_ratio = fuzz.ratio(norm_a, norm_b) / 100.0
    jaro_winkler = JaroWinkler.normalized_similarity(norm_a, norm_b)
    token_sort_ratio = fuzz.token_sort_ratio(norm_a, norm_b) / 100.0
    token_set_ratio = fuzz.token_set_ratio(norm_a, norm_b) / 100.0
    partial_ratio = fuzz.partial_ratio(norm_a, norm_b) / 100.0

    # Phonetic matching - catches typos and transcription errors
    # Soundex and Metaphone are English phonetic algorithms: they produce meaningless
    # codes for non-Latin characters (CJK, Arabic, Cyrillic, etc.), creating noisy
    # false signals. Return NaN when either name contains non-Latin characters
    # so XGBoost learns to ignore phonetics for these pairs.
    either_non_latin = _has_non_latin_alpha(norm_a) or _has_non_latin_alpha(norm_b)

    if either_non_latin:
        soundex_match = _nan
        metaphone_similarity = _nan
    else:
        # Soundex on the key content word (longest word after filtering road type
        # words like street/avenue/north derived from STREET_ABBREVIATIONS).
        key_a = _soundex_key_word(norm_a)
        key_b = _soundex_key_word(norm_b)
        soundex_a = jellyfish.soundex(key_a) if key_a else ""
        soundex_b = jellyfish.soundex(key_b) if key_b else ""
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
        "has_name_ref": has_name_ref,
        "has_name_target": has_name_target,
        "name_is_generic": name_is_generic,
    }


def _extract_all_name_variants(names_dict) -> list[str]:
    """Extract all name variants from an Overture names dict.

    Overture Names schema (https://docs.overturemaps.org/schema/reference/transportation/segment/):
    - primary: The default/main name (string)
    - common: Dict of language code -> name (e.g., {"en": "Lake Geneva", "fr": "Le Léman"})
    - rules: Array of NameRule dicts, each with:
        - value: The name string
        - variant: Type (common, official, alternate, short)
        - language: Language code or None
        - between: [start, end] geometric scope (0-1 fractions along segment)
        - side: Which side of the road (left/right)

    Note: This function extracts ALL name variants regardless of their ``between``
    range or ``side`` scope. The caller (resolve_best_name_variant) uses this for
    cross-language fallback matching — the LR resolution in parse_names_lr already
    handles range-specific name selection for the primary comparison.

    Returns a deduplicated (case-insensitive) list of all available name strings.
    """
    if not names_dict or not isinstance(names_dict, dict):
        return []

    variants: list[str] = []
    seen: set[str] = set()

    def _add(value):
        if isinstance(value, str) and value:
            lower = value.lower()
            if lower not in seen:
                variants.append(value)
                seen.add(lower)

    # 1. Primary name
    _add(names_dict.get("primary"))

    # 2. Common names (dict of language code -> name)
    common = names_dict.get("common")
    if isinstance(common, dict):
        for lang_name in common.values():
            _add(lang_name)

    # 3. Rules (array of NameRule dicts with value, variant, language, between, side)
    rules = names_dict.get("rules")
    if rules is None:
        return variants

    # Handle numpy arrays
    if hasattr(rules, "tolist"):
        rules = rules.tolist()

    if not isinstance(rules, list):
        return variants

    for rule in rules:
        if not isinstance(rule, dict):
            continue
        _add(rule.get("value"))

    return variants


def resolve_best_name_variant(
    ref_names_raw,
    ref_name: str | None,
    target_name: str | None,
) -> str | None:
    """Find the best-matching reference name variant for the target name.

    When Overture has multilingual names (e.g., Chinese primary + English alt),
    the LR-resolved ref_name may be in a different script from the target name,
    yielding ~0 similarity. This function tries ALL available name variants from
    the Overture names dict and returns the highest-scoring one against the target.

    Args:
        ref_names_raw: Raw Overture names dict with primary + rules, or None
        ref_name: Currently resolved reference name (may be None)
        target_name: Target segment name string (may be None)

    Returns:
        The best-matching name variant string, or ref_name if no variants available.
    """
    if not target_name or not ref_names_raw:
        return ref_name

    # Extract target string from dict if needed
    target_str = _extract_name_string(target_name)
    if not target_str:
        return ref_name

    # Extract all available name variants from the reference
    variants = _extract_all_name_variants(ref_names_raw)
    if not variants:
        return ref_name

    # Single variant: return it directly (may upgrade a None ref_name)
    if len(variants) == 1:
        return variants[0]

    norm_target = _normalize_street_name(target_str)
    if not norm_target:
        return ref_name

    best_score = -1.0
    best_variant = ref_name
    for variant in variants:
        norm_variant = _normalize_street_name(variant)
        if not norm_variant:
            continue
        score = fuzz.ratio(norm_variant, norm_target) / 100.0
        if score > best_score:
            best_score = score
            best_variant = variant
            if score == 1.0:
                break  # Perfect match, no need to continue

    return best_variant


def _normalize_street_name(name: str) -> str:
    """Normalize street name for comparison.

    - Unicode NFKC normalization (full-width → half-width, compatibility chars)
    - Convert to lowercase
    - Expand abbreviations (Latin text only)
    - Remove extra whitespace
    - Remove common punctuation
    """
    if not name or not isinstance(name, str):
        return ""

    # Unicode NFKC normalization: normalizes full-width chars (Ｔｏｋｙｏ → Tokyo),
    # compatibility characters, and composed forms. Critical for CJK data where
    # full-width Latin and half-width katakana are common.
    name = unicodedata.normalize("NFKC", name)

    # Lowercase
    name = name.lower().strip()

    # Remove common punctuation
    name = name.replace(".", "").replace(",", "").replace("-", " ")

    # Add spaces around name for abbreviation matching
    name = f" {name} "

    # Expand abbreviations (only effective for Latin text, harmless for CJK)
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

    Uses a two-level scoring system:
    1. Tier check: Are both segments for the same traffic type?
    2. Rank check: Within the same tier, how close are the ranks?

    Traffic tiers:
    - vehicle: motorway, trunk, primary, secondary, tertiary, residential, etc.
    - bicycle: cycleway
    - pedestrian: footway, sidewalk, path, pedestrian, steps
    - neutral: bridleway (uncommon, treated neutrally)

    Cross-tier penalties:
    - vehicle↔pedestrian: 0.1 (strong - cars don't belong on sidewalks)
    - vehicle↔bicycle: 0.7 (mild - bike lanes often on roads)
    - bicycle↔pedestrian: 0.5 (moderate - shared paths exist)

    Args:
        class_a: First road class
        class_b: Second road class
        subclass_a: First subclass (e.g., sidewalk, crosswalk)
        subclass_b: Second subclass

    Returns:
        Similarity score (0-1)
    """
    if not class_a or not isinstance(class_a, str) or not class_b or not isinstance(class_b, str):
        return float("nan")  # Missing data — unknown similarity

    class_a = class_a.lower().strip()
    class_b = class_b.lower().strip()

    # Treat "unknown" class string as missing data
    if class_a == "unknown" or class_b == "unknown":
        return float("nan")

    # Get traffic tiers for both classes
    tier_a = get_traffic_tier(class_a)
    tier_b = get_traffic_tier(class_b)

    # Neutral tier (bridleway) or unknown tier -> no signal
    if tier_a == "neutral" or tier_b == "neutral" or tier_a is None or tier_b is None:
        return float("nan")

    # Cross-tier: lookup penalty from matrix
    if tier_a != tier_b:
        return TIER_PENALTIES.get((tier_a, tier_b), float("nan"))

    # Same tier: use rank-based similarity

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

    # Get hierarchy ranks (default to residential level for unknown classes)
    rank_a = ROAD_CLASS_HIERARCHY.get(class_a, 6)
    rank_b = ROAD_CLASS_HIERARCHY.get(class_b, 6)

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
    if not class_a or not isinstance(class_a, str) or not class_b or not isinstance(class_b, str):
        return True  # Unknown is compatible with anything

    class_a = class_a.lower().strip()
    class_b = class_b.lower().strip()

    if class_a == class_b:
        return True

    # Get hierarchy ranks (default to residential level for unknown classes)
    rank_a = ROAD_CLASS_HIERARCHY.get(class_a, 6)
    rank_b = ROAD_CLASS_HIERARCHY.get(class_b, 6)

    # Allow up to 2 levels difference
    return abs(rank_a - rank_b) <= 2


def get_class_info(class_value: str | None) -> dict:
    """Get diagnostic info about a road class value.

    Returns dict with:
    - normalized: Lowercase, stripped class value
    - known: Whether it's in the hierarchy
    - rank: Hierarchy rank (or default if unknown)

    Useful for debugging class similarity issues.
    """
    if not class_value or not isinstance(class_value, str):
        return {"normalized": None, "known": False, "rank": None}

    normalized = class_value.lower().strip()
    known = normalized in ROAD_CLASS_HIERARCHY
    rank = ROAD_CLASS_HIERARCHY.get(normalized, 6)

    return {"normalized": normalized, "known": known, "rank": rank}


def extract_numeric_suffix(name: str | None) -> int | None:
    """Extract numeric suffix from road name (e.g., 'Interstate 5' -> 5).

    Useful for matching numbered routes.
    """
    if not name:
        return None

    # Find all numbers in the name
    numbers = re.findall(r"\d+", name)

    if numbers:
        # Return the last number (often the route number)
        return int(numbers[-1])

    return None


# Route prefix patterns for canonicalizing route names
# Each pattern maps to a route type. Patterns are checked in order.
ROUTE_PREFIX_PATTERNS = [
    (r"^i[\-\s]?(\d)", "interstate"),
    (r"^interstate\s*", "interstate"),
    (r"^us[\-\s]?(\d)", "us_route"),
    (r"^u\.?s\.?\s*(route|highway|hwy)?\s*", "us_route"),
    (r"^sr[\-\s]?(\d)", "state_route"),
    (r"^state\s*(route|highway|road|rd)\s*", "state_route"),
    (r"^cr[\-\s]?(\d)", "county_road"),
    (r"^county\s*(road|route|rd)\s*", "county_road"),
    (r"^hwy[\-\s]?(\d)", "highway"),
    (r"^highway\s*", "highway"),
]

# Pre-compile patterns for efficiency
_ROUTE_PREFIX_PATTERNS_COMPILED = [
    (re.compile(p, re.IGNORECASE), t) for p, t in ROUTE_PREFIX_PATTERNS
]


def canonicalize_route_name(name: str | None) -> tuple[str | None, int | None]:
    """Canonicalize a route name to (prefix_type, route_number).

    Recognizes common route naming conventions:
    - Interstate: I-5, I 5, Interstate 5
    - US Route: US-101, US 101, U.S. Route 101, US Highway 101
    - State Route: SR-99, SR 99, State Route 99, State Highway 99
    - County Road: CR-15, CR 15, County Road 15
    - Highway: Hwy 1, Highway 1

    Args:
        name: Route name to canonicalize

    Returns:
        Tuple of (prefix_type, route_number) where:
        - prefix_type: "interstate", "us_route", "state_route", "county_road", "highway", or None
        - route_number: The numeric route number, or None if not found
    """
    if not name or not isinstance(name, str):
        return None, None

    name_lower = name.lower().strip()

    # Try each pattern
    for pattern, route_type in _ROUTE_PREFIX_PATTERNS_COMPILED:
        if pattern.search(name_lower):
            # Extract the route number
            route_num = extract_numeric_suffix(name)
            return route_type, route_num

    return None, None


def compute_route_prefix_match(name_a, name_b) -> float:
    """Compute route prefix type matching score.

    Compares the route prefix types (Interstate, US Route, State Route, etc.)
    between two road names. This helps distinguish between different route
    systems that may have the same number (e.g., I-5 vs US-5 vs SR-5).

    Args:
        name_a: First name (string or dict with 'primary' key)
        name_b: Second name (string or dict with 'primary' key)

    Returns:
        1.0 if both have the same route prefix type
        0.0 if both have different route prefix types
        NaN if either/both is not a recognized route (missing signal)
    """
    # Extract name strings from dict if needed
    name_a = _extract_name_string(name_a)
    name_b = _extract_name_string(name_b)

    # Get route prefix types
    prefix_a, _ = canonicalize_route_name(name_a)
    prefix_b, _ = canonicalize_route_name(name_b)

    # Neither is a route - no signal
    if prefix_a is None and prefix_b is None:
        return float("nan")

    # Only one is a route - no signal
    if prefix_a is None or prefix_b is None:
        return float("nan")

    # Both are routes - compare types
    if prefix_a == prefix_b:
        return 1.0
    else:
        return 0.0


def compute_name_numeric_match(name_a, name_b) -> float:
    """Compute numeric route matching score for numbered routes (I-90, US-101, etc.).

    This feature helps match numbered routes that may have different formatting
    (e.g., "Interstate 90" vs "I-90", "US Route 101" vs "US-101").

    Args:
        name_a: First name (string or dict with 'primary' key)
        name_b: Second name (string or dict with 'primary' key)

    Returns:
        1.0 if both have matching route numbers
        NaN if neither has a number (no signal)
        0.0 if route numbers mismatch
        NaN if only one has a number (no signal)
    """
    # Extract name strings from dict if needed
    name_a = _extract_name_string(name_a)
    name_b = _extract_name_string(name_b)

    # Extract numeric suffixes
    num_a = extract_numeric_suffix(name_a)
    num_b = extract_numeric_suffix(name_b)

    # Neither has a number - no signal
    if num_a is None and num_b is None:
        return float("nan")

    # Only one has a number - no signal
    if num_a is None or num_b is None:
        return float("nan")

    # Both have numbers - check if they match
    if num_a == num_b:
        return 1.0
    else:
        return 0.0


def names_likely_same_road(name_a: str | None, name_b: str | None) -> bool:
    """Quick check if two names likely refer to the same road.

    Uses multiple heuristics for a quick yes/no decision.
    """
    if not name_a or not isinstance(name_a, str) or not name_b or not isinstance(name_b, str):
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
