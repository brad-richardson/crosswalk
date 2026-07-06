/**
 * Dataset browser for the matcher web UI.
 *
 * Loads and displays raw fetched features on the map for any dataset.
 * Uses the persistent MapLibre map from map.js.
 */
(function () {
    "use strict";

    var map = window.matcherMap;
    var geojsonBounds = window.matcherGeojsonBounds;

    var BROWSER_SOURCE = "browser-features";
    var BROWSER_LAYER = "browser-features-line";
    var popup = null;

    // Class-based color scheme
    var classColors = {
        cycleway: "#2e7d32",
        footway: "#e65100",
        path: "#6a1b9a",
        pedestrian: "#ef6c00",
        residential: "#1565c0",
        tertiary: "#1976d2",
        secondary: "#0d47a1",
        primary: "#b71c1c",
        trunk: "#880e4f",
        motorway: "#4a148c",
        service: "#78909c",
        unclassified: "#607d8b",
        unknown: "#9e9e9e",
    };

    function escapeHtml(str) {
        if (!str) return "";
        var div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    function clearBrowserLayer() {
        if (popup) { popup.remove(); popup = null; }
        if (map.getLayer(BROWSER_LAYER)) map.removeLayer(BROWSER_LAYER);
        if (map.getSource(BROWSER_SOURCE)) map.removeSource(BROWSER_SOURCE);
    }

    function showInfo(metadata) {
        var infoEl = document.getElementById("browser-info");
        var nameEl = document.getElementById("browser-dataset-name");
        var statsEl = document.getElementById("browser-stats");

        if (!infoEl) return;

        nameEl.textContent = metadata.display_name || metadata.dataset;

        var parts = [];
        parts.push(metadata.total_count.toLocaleString() + " features");

        var types = metadata.geometry_types || {};
        var typeList = Object.keys(types).map(function (k) {
            return k + ": " + types[k];
        });
        if (typeList.length > 0) {
            parts.push(typeList.join(", "));
        }

        if (metadata.truncated) {
            parts.push("<em>Showing first " + metadata.returned_count.toLocaleString() + "</em>");
        }

        statsEl.innerHTML = parts.join("<br>");
        infoEl.classList.remove("hidden");
    }

    function addBrowserLayer(data) {
        // Build a match expression for class → color
        var colorExpr = ["match", ["coalesce", ["get", "class"], "unknown"]];
        var keys = Object.keys(classColors);
        for (var i = 0; i < keys.length; i++) {
            colorExpr.push(keys[i]);
            colorExpr.push(classColors[keys[i]]);
        }
        colorExpr.push("#9e9e9e"); // fallback

        map.addSource(BROWSER_SOURCE, {
            type: "geojson",
            data: data,
        });

        map.addLayer({
            id: BROWSER_LAYER,
            type: "line",
            source: BROWSER_SOURCE,
            paint: {
                "line-color": colorExpr,
                "line-width": 3,
                "line-opacity": 0.8,
            },
        });

        // Click to show popup
        map.on("click", BROWSER_LAYER, function (e) {
            if (!e.features || !e.features.length) return;

            var props = e.features[0].properties || {};
            var parts = [];

            if (props.name) parts.push("<b>" + escapeHtml(props.name) + "</b>");
            if (props.id) parts.push("<span style='font-family:monospace;font-size:0.8em;color:#888'>" + escapeHtml(props.id) + "</span>");
            if (props["class"]) parts.push("Class: <b>" + escapeHtml(props["class"]) + "</b>");
            if (props.subclass) parts.push("Subclass: " + escapeHtml(props.subclass));

            if (parts.length > 0) {
                if (popup) popup.remove();
                popup = new maplibregl.Popup({ maxWidth: "300px" })
                    .setLngLat(e.lngLat)
                    .setHTML(parts.join("<br>"))
                    .addTo(map);
            }
        });

        // Cursor change on hover
        map.on("mouseenter", BROWSER_LAYER, function () {
            map.getCanvas().style.cursor = "pointer";
        });
        map.on("mouseleave", BROWSER_LAYER, function () {
            map.getCanvas().style.cursor = "";
        });
    }

    function loadDataset(dataset) {
        if (!dataset) return;

        // Show loading
        var infoEl = document.getElementById("browser-info");
        if (infoEl) {
            infoEl.classList.remove("hidden");
            document.getElementById("browser-dataset-name").textContent = "Loading...";
            document.getElementById("browser-stats").textContent = "";
        }

        // Clear existing features
        clearBrowserLayer();

        fetch("/browser/features?dataset=" + encodeURIComponent(dataset))
            .then(function (resp) {
                if (!resp.ok) throw new Error("Failed to load features: " + resp.status);
                return resp.json();
            })
            .then(function (data) {
                var metadata = data.metadata || {};

                // Remove metadata before passing to MapLibre
                delete data.metadata;

                // Wait for style to be loaded before adding layers
                function doAdd() {
                    addBrowserLayer(data);

                    var bbox = geojsonBounds(data);
                    if (bbox) {
                        map.fitBounds(bbox, { padding: 40 });
                    }

                    showInfo(metadata);
                }

                if (map.isStyleLoaded()) {
                    doAdd();
                } else {
                    map.on("load", doAdd);
                }
            })
            .catch(function (err) {
                console.error("Browser load error:", err);
                if (infoEl) {
                    document.getElementById("browser-dataset-name").textContent = "Error";
                    document.getElementById("browser-stats").textContent = err.message;
                }
            });
    }

    // Expose for inline scripts
    window.browserLoadDataset = loadDataset;
})();
