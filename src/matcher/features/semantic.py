"""Semantic feature extraction (names, classifications)."""

import re

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
    if not road_class:
        return None
    return TRAFFIC_TIERS.get(road_class.lower().strip())


def compute_tier_match(ref_class: str | None, target_class: str | None) -> float:
    """Compute whether reference and target are in the same traffic tier.

    This binary feature allows the ML model to learn hard penalties for
    cross-tier mismatches (e.g., residential road matching sidewalk).

    Args:
        ref_class: Reference road class
        target_class: Target road class

    Returns:
        1.0 if same traffic tier, 0.0 if different tiers, 0.5 if either unknown
    """
    ref_tier = get_traffic_tier(ref_class)
    target_tier = get_traffic_tier(target_class)

    # Unknown tier -> neutral score
    if ref_tier is None or target_tier is None:
        return 0.5

    # Neutral tier -> neutral score
    if ref_tier == "neutral" or target_tier == "neutral":
        return 0.5

    return 1.0 if ref_tier == target_tier else 0.0


def compute_tier_incompatible(ref_class: str | None, target_class: str | None) -> float:
    """Compute whether reference and target are vehicle↔pedestrian mismatch.

    This specific binary feature flags the most problematic cross-tier matches:
    vehicular roads matched with pedestrian infrastructure (sidewalks, footways).

    Args:
        ref_class: Reference road class
        target_class: Target road class

    Returns:
        1.0 if vehicle↔pedestrian mismatch, 0.0 otherwise
    """
    ref_tier = get_traffic_tier(ref_class)
    target_tier = get_traffic_tier(target_class)

    if ref_tier is None or target_tier is None:
        return 0.0

    return 1.0 if {ref_tier, target_tier} == {"vehicle", "pedestrian"} else 0.0


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
        "has_name_ref": has_name_ref,
        "has_name_target": has_name_target,
        "name_is_generic": name_is_generic,
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
    if not class_a or not class_b:
        return 0.5  # Unknown, neutral

    class_a = class_a.lower().strip()
    class_b = class_b.lower().strip()

    # Treat "unknown" as neutral - don't penalize or reward
    if class_a == "unknown" or class_b == "unknown":
        return 0.5

    # Get traffic tiers for both classes
    tier_a = get_traffic_tier(class_a)
    tier_b = get_traffic_tier(class_b)

    # Neutral tier (bridleway) or unknown tier -> return 0.5
    if tier_a == "neutral" or tier_b == "neutral" or tier_a is None or tier_b is None:
        return 0.5

    # Cross-tier: lookup penalty from matrix
    if tier_a != tier_b:
        return TIER_PENALTIES.get((tier_a, tier_b), 0.5)

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
    if not class_a or not class_b:
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
    if not class_value:
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
    if not name:
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
        0.5 if either/both is not a recognized route (neutral)
    """
    # Extract name strings from dict if needed
    name_a = _extract_name_string(name_a)
    name_b = _extract_name_string(name_b)

    # Get route prefix types
    prefix_a, _ = canonicalize_route_name(name_a)
    prefix_b, _ = canonicalize_route_name(name_b)

    # Neither is a route - neutral (no signal)
    if prefix_a is None and prefix_b is None:
        return 0.5

    # Only one is a route - neutral (don't penalize)
    if prefix_a is None or prefix_b is None:
        return 0.5

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
        0.5 if neither has a number (neutral - no signal either way)
        0.0 if route numbers mismatch
        0.5 if only one has a number (neutral - don't penalize)
    """
    # Extract name strings from dict if needed
    name_a = _extract_name_string(name_a)
    name_b = _extract_name_string(name_b)

    # Extract numeric suffixes
    num_a = extract_numeric_suffix(name_a)
    num_b = extract_numeric_suffix(name_b)

    # Neither has a number - neutral (no signal either way)
    if num_a is None and num_b is None:
        return 0.5

    # Only one has a number - neutral (don't penalize)
    if num_a is None or num_b is None:
        return 0.5

    # Both have numbers - check if they match
    if num_a == num_b:
        return 1.0
    else:
        return 0.0


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


def compute_oneway_match(
    oneway_ref: str | None,
    oneway_target: str | None,
) -> float:
    """Compute one-way direction compatibility (0-1).

    One-way values:
    - "forward": One-way in the direction of digitization
    - "backward": One-way against the direction of digitization
    - "both": Bidirectional (two-way)
    - None: Unknown

    Scoring:
    - Same direction (both "forward", both "backward", both "both"): 1.0
    - Opposite direction (forward vs backward): 0.1 (strong mismatch)
    - One bidirectional, one one-way: 0.5 (could be valid)
    - Either missing: 0.5 (neutral - no signal)

    Args:
        oneway_ref: Reference one-way direction
        oneway_target: Target one-way direction

    Returns:
        Compatibility score (0.0-1.0)
    """
    if not oneway_ref or not oneway_target:
        return 0.5

    if oneway_ref == oneway_target:
        return 1.0

    # Opposite directions = strong mismatch
    if {oneway_ref, oneway_target} == {"forward", "backward"}:
        return 0.1

    # One bidirectional, one one-way = moderate uncertainty
    return 0.5


def compute_speed_limit_similarity(
    speed_ref: int | None,
    speed_target: int | None,
) -> float:
    """Compute speed limit similarity (0-1).

    Uses min/max ratio for a smooth similarity score.
    Assumes both values are in the same unit (kph).

    Scoring:
    - Exact match: 1.0
    - Similar speeds: ratio (e.g., 50/60 = 0.83)
    - Either missing: 0.5 (neutral - no signal)
    - Either invalid (<=0): 0.5 (neutral)

    Args:
        speed_ref: Reference speed limit in kph
        speed_target: Target speed limit in kph

    Returns:
        Similarity score (0.0-1.0)
    """
    if speed_ref is None or speed_target is None:
        return 0.5
    if speed_ref <= 0 or speed_target <= 0:
        return 0.5

    return min(speed_ref, speed_target) / max(speed_ref, speed_target)
