/**
 * QA map layer for interactive edge review.
 *
 * Renders integration edges on the MapLibre map with color-coded layers.
 * Non-reference edges are clickable and fetch edge detail via HTMX.
 */
(function () {
    "use strict";

    var map = window.matcherMap;
    var geojsonBounds = window.matcherGeojsonBounds;

    if (!map) {
        console.error("qa-map.js: matcherMap not available");
        return;
    }

    // Clear existing pair source data
    var pairSource = window.matcherPairLayer;
    if (pairSource && map.getSource(pairSource)) {
        map.getSource(pairSource).setData({ type: "FeatureCollection", features: [] });
    }

    if (typeof edgeGeojson === "undefined" || !edgeGeojson || !edgeGeojson.features) {
        return;
    }

    var QA_SOURCE = "qa-edges";
    var QA_REF_LAYER = "qa-edges-reference";
    var QA_LAYER = "qa-edges-interactive";
    var selectedEdgeId = null;

    function setup() {
        // Remove previous QA layers/source if they exist
        [QA_LAYER, QA_REF_LAYER].forEach(function (id) {
            if (map.getLayer(id)) map.removeLayer(id);
        });
        if (map.getSource(QA_SOURCE)) map.removeSource(QA_SOURCE);

        // Tag features with _isReference for filtering
        var features = edgeGeojson.features.map(function (f) {
            var props = Object.assign({}, f.properties || {});
            props._isReference = props.layer === "reference";
            return { type: "Feature", geometry: f.geometry, properties: props };
        });

        map.addSource(QA_SOURCE, {
            type: "geojson",
            data: { type: "FeatureCollection", features: features },
        });

        // Reference layer (underneath, non-interactive)
        map.addLayer({
            id: QA_REF_LAYER,
            type: "line",
            source: QA_SOURCE,
            filter: ["==", ["get", "_isReference"], true],
            paint: {
                "line-color": ["coalesce", ["get", "color"], "#999"],
                "line-width": 2,
                "line-opacity": 0.4,
            },
        });

        // Interactive (non-reference) layer
        map.addLayer({
            id: QA_LAYER,
            type: "line",
            source: QA_SOURCE,
            filter: ["==", ["get", "_isReference"], false],
            paint: {
                "line-color": ["coalesce", ["get", "color"], "#999"],
                "line-width": [
                    "case",
                    ["==", ["get", "edge_id"], selectedEdgeId || ""],
                    6,
                    3,
                ],
                "line-opacity": 0.8,
            },
        });

        // Click handler for non-reference edges
        map.on("click", QA_LAYER, function (e) {
            if (!e.features || !e.features.length) return;

            var feat = e.features[0];
            var edgeId = feat.properties.edge_id;
            selectedEdgeId = edgeId;

            // Update line widths to highlight selection
            map.setPaintProperty(QA_LAYER, "line-width", [
                "case",
                ["==", ["get", "edge_id"], edgeId],
                6,
                3,
            ]);

            // Fetch edge detail via HTMX
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

        // Change cursor on hover
        map.on("mouseenter", QA_LAYER, function () {
            map.getCanvas().style.cursor = "pointer";
        });
        map.on("mouseleave", QA_LAYER, function () {
            map.getCanvas().style.cursor = "";
        });

        // Fit bounds to non-reference features
        var nonRefFC = {
            type: "FeatureCollection",
            features: features.filter(function (f) { return !f.properties._isReference; }),
        };
        var bbox = geojsonBounds(nonRefFC);
        if (bbox) {
            map.fitBounds(bbox, { padding: 60 });
        }
    }

    // Run setup after style is loaded (in case map.js style.load hasn't fired yet)
    if (map.isStyleLoaded()) {
        setup();
    } else {
        map.on("load", setup);
    }
})();
