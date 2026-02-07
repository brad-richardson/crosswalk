/**
 * Persistent Leaflet map for the matcher web UI.
 *
 * Creates a fullscreen map centered on Boston with multiple tile layers,
 * a layer group for pair geometries, and HTMX integration for dynamic
 * content loading.
 */
(function () {
    "use strict";

    // --- Map initialization ---
    var map = L.map("map", {
        center: [42.36, -71.06],
        zoom: 14,
        zoomControl: false,
    });

    // Add zoom control to bottom-left
    L.control.zoom({ position: "bottomleft" }).addTo(map);

    // --- Tile layers ---
    var light = L.tileLayer(
        "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png",
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OSM</a> &copy; <a href="https://carto.com/">CARTO</a>',
            subdomains: "abcd",
            maxZoom: 20,
        }
    );

    var satellite = L.tileLayer(
        "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
        {
            attribution: "&copy; Esri",
            maxZoom: 19,
        }
    );

    var osm = L.tileLayer(
        "https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png",
        {
            attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>',
            maxZoom: 19,
        }
    );

    // Default to light tiles
    light.addTo(map);

    // Layer control
    L.control
        .layers(
            { Light: light, Satellite: satellite, OSM: osm },
            {},
            { position: "bottomleft" }
        )
        .addTo(map);

    // --- Pair geometry layer group ---
    var pairLayer = L.featureGroup().addTo(map);

    // Style definitions for geometry types
    var styles = {
        reference: {
            color: "#2196F3",
            weight: 4,
            opacity: 0.9,
        },
        target: {
            color: "#FF5722",
            weight: 4,
            opacity: 0.9,
            dashArray: "8, 6",
        },
        referenceFull: {
            color: "#2196F3",
            weight: 2,
            opacity: 0.3,
        },
        targetFull: {
            color: "#FF5722",
            weight: 2,
            opacity: 0.3,
            dashArray: "8, 6",
        },
    };

    /**
     * Display pair geometries on the map from a data-geometry attribute.
     *
     * Expected JSON structure:
     * {
     *   "reference": <GeoJSON>,
     *   "target": <GeoJSON>,
     *   "reference_full": <GeoJSON> (optional),
     *   "target_full": <GeoJSON> (optional)
     * }
     */
    function showPairGeometry(geojsonData) {
        pairLayer.clearLayers();

        if (!geojsonData) return;

        var data;
        if (typeof geojsonData === "string") {
            try {
                data = JSON.parse(geojsonData);
            } catch (e) {
                console.error("Failed to parse geometry data:", e);
                return;
            }
        } else {
            data = geojsonData;
        }

        // Add full geometries first (underneath, faded)
        if (data.reference_full) {
            L.geoJSON(data.reference_full, { style: styles.referenceFull }).addTo(pairLayer);
        }
        if (data.target_full) {
            L.geoJSON(data.target_full, { style: styles.targetFull }).addTo(pairLayer);
        }

        // Add primary geometries on top
        if (data.reference) {
            L.geoJSON(data.reference, { style: styles.reference }).addTo(pairLayer);
        }
        if (data.target) {
            L.geoJSON(data.target, { style: styles.target }).addTo(pairLayer);
        }

        // Fit map bounds to the aligned sublines (not full segments)
        var sublineBounds = L.featureGroup();
        if (data.reference) {
            L.geoJSON(data.reference).addTo(sublineBounds);
        }
        if (data.target) {
            L.geoJSON(data.target).addTo(sublineBounds);
        }
        var bounds = sublineBounds.getBounds();
        if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [60, 60] });
        }
    }

    /**
     * Read geometry from a <script type="application/json" id="pair-geometry">
     * element and render it on the map.
     */
    function loadPairGeometry() {
        var el = document.getElementById("pair-geometry");
        if (el) {
            try {
                var data = JSON.parse(el.textContent);
                showPairGeometry(data);
            } catch (e) {
                console.error("Failed to parse pair geometry:", e);
            }
        }
    }

    // --- Initial load ---
    loadPairGeometry();

    // --- HTMX integration ---
    // After HTMX swaps in new content, re-read the geometry data
    document.addEventListener("htmx:afterSwap", function () {
        loadPairGeometry();
    });

    // Expose map and helpers for console debugging
    window.matcherMap = map;
    window.matcherPairLayer = pairLayer;
    window.matcherShowGeometry = showPairGeometry;
})();
