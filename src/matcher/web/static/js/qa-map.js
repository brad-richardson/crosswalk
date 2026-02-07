/**
 * QA map layer for interactive edge review.
 *
 * Renders integration edges on the Leaflet map with color-coded layers.
 * Non-reference edges are clickable and fetch edge detail via HTMX.
 */
(function () {
    "use strict";

    var map = window.matcherMap;
    var pairLayer = window.matcherPairLayer;

    if (!map || !pairLayer) {
        console.error("qa-map.js: matcherMap or matcherPairLayer not available");
        return;
    }

    // Clear existing layers
    pairLayer.clearLayers();

    // Track the currently selected layer for highlight
    var selectedLayer = null;

    if (typeof edgeGeojson === "undefined" || !edgeGeojson || !edgeGeojson.features) {
        return;
    }

    var nonRefBounds = L.latLngBounds();

    edgeGeojson.features.forEach(function (feature) {
        var props = feature.properties || {};
        var isReference = props.layer === "reference";

        var style = {
            color: props.color || "#999",
            weight: isReference ? 2 : 3,
            opacity: isReference ? 0.4 : 0.8,
            interactive: !isReference,
        };

        var layer = L.geoJSON(feature, {
            style: function () {
                return style;
            },
            interactive: !isReference,
            onEachFeature: function (feat, lyr) {
                if (!isReference) {
                    // Track bounds for non-reference edges
                    var bounds = lyr.getBounds();
                    if (bounds.isValid()) {
                        nonRefBounds.extend(bounds);
                    }

                    lyr.on("click", function () {
                        // Reset previously selected layer
                        if (selectedLayer) {
                            selectedLayer.setStyle({ weight: 3 });
                        }

                        // Highlight clicked layer
                        lyr.setStyle({ weight: 6 });
                        selectedLayer = lyr;

                        // Fetch edge detail via HTMX
                        var edgeId = feat.properties.edge_id;
                        var url =
                            "/qa/edge/" +
                            edgeId +
                            "?dataset=" +
                            encodeURIComponent(dataset) +
                            "&type=" +
                            encodeURIComponent(edgeType);

                        htmx.ajax("GET", url, {
                            target: "#edge-detail",
                            swap: "innerHTML",
                        });
                    });
                }
            },
        });

        layer.addTo(pairLayer);
    });

    // Fit map bounds to non-reference edges
    if (nonRefBounds.isValid()) {
        map.fitBounds(nonRefBounds, { padding: [60, 60] });
    }
})();
