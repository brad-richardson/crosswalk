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
    var pairLayer = L.layerGroup().addTo(map);

    // Style definitions for geometry types
    var styles = {
        gers: {
            color: "#2196F3",
            weight: 4,
            opacity: 0.9,
        },
        local: {
            color: "#FF5722",
            weight: 4,
            opacity: 0.9,
            dashArray: "8, 6",
        },
        gersFull: {
            color: "#2196F3",
            weight: 2,
            opacity: 0.3,
        },
        localFull: {
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
     *   "gers": <GeoJSON>,
     *   "local": <GeoJSON>,
     *   "gers_full": <GeoJSON> (optional),
     *   "local_full": <GeoJSON> (optional)
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
        if (data.gers_full) {
            L.geoJSON(data.gers_full, { style: styles.gersFull }).addTo(pairLayer);
        }
        if (data.local_full) {
            L.geoJSON(data.local_full, { style: styles.localFull }).addTo(pairLayer);
        }

        // Add primary geometries on top
        if (data.gers) {
            L.geoJSON(data.gers, { style: styles.gers }).addTo(pairLayer);
        }
        if (data.local) {
            L.geoJSON(data.local, { style: styles.local }).addTo(pairLayer);
        }

        // Fit map bounds to the pair with padding
        var bounds = pairLayer.getBounds();
        if (bounds.isValid()) {
            map.fitBounds(bounds, { padding: [60, 60] });
        }
    }

    // --- Initial geometry load ---
    // Read geometry from the page on first load (before any HTMX swaps)
    document.addEventListener("DOMContentLoaded", function () {
        var geomEl = document.querySelector("[data-geometry]");
        if (geomEl) {
            showPairGeometry(geomEl.dataset.geometry);
        }
    });

    // --- HTMX integration ---
    // After HTMX swaps in new content, check for geometry data and display it
    document.addEventListener("htmx:afterSwap", function (event) {
        var target = event.detail.target;
        if (!target) return;

        // Look for data-geometry attribute in the swapped content
        var geometryEl = target.querySelector("[data-geometry]");
        if (geometryEl) {
            var geojsonStr = geometryEl.getAttribute("data-geometry");
            showPairGeometry(geojsonStr);
        }
    });

    // Expose map and helpers for console debugging
    window.matcherMap = map;
    window.matcherPairLayer = pairLayer;
    window.matcherShowGeometry = showPairGeometry;
})();
