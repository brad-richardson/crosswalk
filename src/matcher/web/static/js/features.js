/**
 * Feature quality signal constants and classification logic.
 * Shared across labeling and batch label modes.
 */

// "higher is better" features (similarity, coverage, match scores: 0-1 scale)
var HIGHER_BETTER = [
    'name_levenshtein', 'name_jaro_winkler', 'name_token_sort',
    'name_soundex', 'name_metaphone', 'name_numeric_match', 'route_prefix_match',
    'class_similarity', 'buffer_iou_5m', 'buffer_iou_15m',
    'ref_coverage', 'target_coverage', 'min_coverage', 'coverage_ratio',
    'angle_histogram_similarity', 'degree_match_score', 'degree_signature_similarity',
    'graphlet_similarity', 'endpoint_degree_similarity',
    'shared_endpoint_count', 'dead_end_match', 'intersection_match',
    'has_name_ref', 'has_name_target',
];

// "lower is better" features (distances, deltas)
var LOWER_BETTER = [
    'hausdorff_distance_m', 'mean_hausdorff_distance_m', 'hausdorff_p95_m',
    'lateral_offset_m', 'lateral_offset_iqr_m',
    'lateral_offset_p95_m', 'edge_distance_rmse_m',
    'min_endpoint_proximity_m', 'max_endpoint_proximity_m',
    'heading_delta', 'sinuosity_delta', 'heading_consistency_delta',
    'clustering_coef_delta', 'collinear_gap_ratio',
];

// Error/default sentinel values for UI display (subset of _get_error_features() in compute.py)
var MAX_DIST = 10000;
var ERROR_DEFAULTS = {
    'hausdorff_distance_m': MAX_DIST, 'mean_hausdorff_distance_m': MAX_DIST,
    'hausdorff_p95_m': MAX_DIST,
    'edge_distance_rmse_m': MAX_DIST,
    'lateral_offset_m': MAX_DIST, 'lateral_offset_iqr_m': MAX_DIST,
    'lateral_offset_p95_m': MAX_DIST,
    'min_endpoint_proximity_m': MAX_DIST, 'max_endpoint_proximity_m': MAX_DIST,
    'heading_delta': 180, 'collinear_gap_ratio': 1.0,
};

function classifyFeature(name, value) {
    var v = parseFloat(value);
    if (isNaN(v)) return '';

    // Check if this looks like an error default
    if (ERROR_DEFAULTS[name] !== undefined && v === ERROR_DEFAULTS[name]) {
        return 'error';
    }

    if (HIGHER_BETTER.indexOf(name) !== -1) {
        // 0-1 scale: >= 0.7 good, 0.3-0.7 mid, < 0.3 bad
        if (v >= 0.7) return 'good';
        if (v >= 0.3) return 'mid';
        return 'bad';
    }
    if (LOWER_BETTER.indexOf(name) !== -1) {
        // Distance features: < 10m good, 10-50m mid, > 50m bad
        // Angle/ratio features: < 15 good, 15-45 mid, > 45 bad
        var isAngle = name.indexOf('heading') !== -1 || name.indexOf('delta') !== -1;
        var isRatio = name.indexOf('ratio') !== -1;
        if (isAngle) {
            if (v <= 15) return 'good';
            if (v <= 45) return 'mid';
            return 'bad';
        }
        if (isRatio) {
            if (v <= 0.1) return 'good';
            if (v <= 0.3) return 'mid';
            return 'bad';
        }
        // Distance in meters
        if (v <= 10) return 'good';
        if (v <= 50) return 'mid';
        return 'bad';
    }
    return '';  // neutral for unclassified features
}

function applyFeatureSignals() {
    document.querySelectorAll('.feature-signal').forEach(function(el) {
        var name = el.dataset.feature;
        var value = el.dataset.value;
        var cls = classifyFeature(name, value);
        if (cls === 'good') el.innerHTML = '<span class="signal signal-good" title="Good"></span>';
        else if (cls === 'mid') el.innerHTML = '<span class="signal signal-mid" title="Moderate"></span>';
        else if (cls === 'bad') el.innerHTML = '<span class="signal signal-bad" title="Poor"></span>';
        else if (cls === 'error') el.innerHTML = '<span class="signal signal-error" title="Error/default value">&#9888;</span>';
    });
}

// Apply on initial load
applyFeatureSignals();

// Re-apply after HTMX swaps (including OOB swaps for features drawer)
document.addEventListener('htmx:afterSettle', function() {
    applyFeatureSignals();
});

// Pair ID expand/collapse (tap to show full ID, tap again to collapse)
document.addEventListener('click', function(e) {
    var el = e.target.closest('.pair-id');
    if (!el) return;
    el.classList.toggle('expanded');
    if (el.classList.contains('expanded')) {
        el.textContent = el.dataset.full;
    } else {
        el.textContent = el.dataset.short;
    }
});
