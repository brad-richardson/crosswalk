/**
 * Audit mode interaction: pair navigation, histograms, category toggles.
 */
(function () {
    "use strict";

    // --- Pair list navigation ---
    window.selectPairRow = function (el) {
        document.querySelectorAll('.audit-pair-row.selected').forEach(function (r) {
            r.classList.remove('selected');
        });
        el.classList.add('selected');
    };

    // Keyboard navigation (up/down arrows)
    document.addEventListener('keydown', function (e) {
        if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
        var rows = document.querySelectorAll('.audit-pair-row');
        if (rows.length === 0) return;

        var current = document.querySelector('.audit-pair-row.selected');
        var idx = current ? Array.prototype.indexOf.call(rows, current) : -1;

        if (e.key === 'ArrowDown' || e.key === 'j') {
            e.preventDefault();
            var next = Math.min(idx + 1, rows.length - 1);
            rows[next].click();
            rows[next].scrollIntoView({ block: 'nearest' });
        } else if (e.key === 'ArrowUp' || e.key === 'k') {
            e.preventDefault();
            var prev = Math.max(idx - 1, 0);
            rows[prev].click();
            rows[prev].scrollIntoView({ block: 'nearest' });
        }
    });

    // --- Sort control ---
    window.auditSort = function (feature) {
        if (!feature) return;
        var params = new URLSearchParams(window.location.search);
        var currentSort = params.get('sort');
        var currentOrder = params.get('order') || 'asc';
        var newOrder = (feature === currentSort && currentOrder === 'asc') ? 'desc' : 'asc';
        params.set('sort', feature);
        params.set('order', newOrder);
        window.location.search = params.toString();
    };

    // --- Category collapse/expand ---
    window.toggleCategory = function (header) {
        var body = header.nextElementSibling;
        var arrow = header.querySelector('.audit-category-arrow');
        if (body.style.display === 'none') {
            body.style.display = '';
            arrow.innerHTML = '&#9660;';
        } else {
            body.style.display = 'none';
            arrow.innerHTML = '&#9654;';
        }
    };

    // --- Histogram rendering ---
    window.loadHistogram = function (dataset, feature, currentValue) {
        var container = document.getElementById('audit-histogram');
        var title = document.getElementById('histogram-title');
        var svgDiv = document.getElementById('histogram-svg');

        title.textContent = feature;
        container.style.display = 'block';
        svgDiv.innerHTML = '<span class="spinner"></span> Loading...';

        fetch('/audit/distributions?dataset=' + encodeURIComponent(dataset) +
              '&feature=' + encodeURIComponent(feature))
            .then(function (r) { return r.json(); })
            .then(function (data) {
                renderHistogram(svgDiv, data, currentValue);
            })
            .catch(function (err) {
                svgDiv.innerHTML = 'Failed to load distribution data.';
            });
    };

    window.closeHistogram = function () {
        document.getElementById('audit-histogram').style.display = 'none';
    };

    function renderHistogram(container, data, currentValue) {
        if (!data.bins || data.bins.length === 0) {
            container.innerHTML = '<em>No data</em>';
            return;
        }

        var w = 280, h = 120;
        var margin = { top: 10, right: 10, bottom: 20, left: 5 };
        var plotW = w - margin.left - margin.right;
        var plotH = h - margin.top - margin.bottom;

        var maxVal = 0;
        for (var i = 0; i < data.match.length; i++) {
            maxVal = Math.max(maxVal, data.match[i], data.no_match[i]);
        }
        if (maxVal === 0) maxVal = 1;

        var barW = plotW / data.bins.length;

        var svg = '<svg width="' + w + '" height="' + h + '">';
        svg += '<g transform="translate(' + margin.left + ',' + margin.top + ')">';

        // Match bars (green, left half of each bin)
        for (var i = 0; i < data.match.length; i++) {
            var bh = (data.match[i] / maxVal) * plotH;
            var x = i * barW;
            var y = plotH - bh;
            svg += '<rect x="' + x + '" y="' + y + '" width="' + (barW * 0.45) +
                   '" height="' + bh + '" fill="#4caf50" opacity="0.7"/>';
        }

        // No-match bars (red, right half of each bin)
        for (var i = 0; i < data.no_match.length; i++) {
            var bh = (data.no_match[i] / maxVal) * plotH;
            var x = i * barW + barW * 0.5;
            var y = plotH - bh;
            svg += '<rect x="' + x + '" y="' + y + '" width="' + (barW * 0.45) +
                   '" height="' + bh + '" fill="#f44336" opacity="0.7"/>';
        }

        // Current value marker
        if (currentValue != null && data.range) {
            var range = data.range[1] - data.range[0];
            if (range > 0) {
                var cx = ((currentValue - data.range[0]) / range) * plotW;
                svg += '<line x1="' + cx + '" y1="0" x2="' + cx + '" y2="' + plotH +
                       '" stroke="#1976d2" stroke-width="2" stroke-dasharray="4,2"/>';
            }
        }

        // Baseline
        svg += '<line x1="0" y1="' + plotH + '" x2="' + plotW + '" y2="' + plotH +
               '" stroke="#ccc" stroke-width="1"/>';

        // Labels
        if (data.range) {
            svg += '<text x="0" y="' + (plotH + 14) + '" font-size="10" fill="#999">' +
                   data.range[0].toFixed(2) + '</text>';
            svg += '<text x="' + plotW + '" y="' + (plotH + 14) +
                   '" font-size="10" fill="#999" text-anchor="end">' +
                   data.range[1].toFixed(2) + '</text>';
        }

        svg += '</g>';

        // Legend
        svg += '<rect x="' + (w - 90) + '" y="2" width="8" height="8" fill="#4caf50" opacity="0.7"/>';
        svg += '<text x="' + (w - 78) + '" y="10" font-size="9" fill="#666">Match</text>';
        svg += '<rect x="' + (w - 90) + '" y="14" width="8" height="8" fill="#f44336" opacity="0.7"/>';
        svg += '<text x="' + (w - 78) + '" y="22" font-size="9" fill="#666">No Match</text>';

        svg += '</svg>';
        container.innerHTML = svg;
    }

    // Re-apply feature signals after HTMX swaps in pair detail
    document.addEventListener('htmx:afterSettle', function () {
        if (typeof applyFeatureSignals === 'function') {
            applyFeatureSignals();
        }
    });
})();
