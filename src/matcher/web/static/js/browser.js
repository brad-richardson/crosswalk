/**
 * Dataset browser for the matcher web UI.
 *
 * Loads and displays raw fetched features on the map for any dataset.
 * Uses the persistent Leaflet map from map.js.
 */
(function () {
    "use strict";

    var map = window.matcherMap;
    var featureLayer = null;

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

    function getColor(cls) {
        return classColors[cls] || "#9e9e9e";
    }

    function styleFeature(feature) {
        var cls = (feature.properties && feature.properties["class"]) || "unknown";
        return {
            color: getColor(cls),
            weight: 3,
            opacity: 0.8,
        };
    }

    function onEachFeature(feature, layer) {
        if (!feature.properties) return;

        var props = feature.properties;
        var parts = [];

        if (props.name) parts.push("<b>" + escapeHtml(props.name) + "</b>");
        if (props.id) parts.push("<span style='font-family:monospace;font-size:0.8em;color:#888'>" + escapeHtml(props.id) + "</span>");
        if (props["class"]) parts.push("Class: <b>" + escapeHtml(props["class"]) + "</b>");
        if (props.subclass) parts.push("Subclass: " + escapeHtml(props.subclass));

        if (parts.length > 0) {
            layer.bindPopup(parts.join("<br>"), { maxWidth: 300 });
        }
    }

    function escapeHtml(str) {
        if (!str) return "";
        var div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
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
        if (featureLayer) {
            map.removeLayer(featureLayer);
            featureLayer = null;
        }

        fetch("/browser/features?dataset=" + encodeURIComponent(dataset))
            .then(function (resp) {
                if (!resp.ok) throw new Error("Failed to load features: " + resp.status);
                return resp.json();
            })
            .then(function (data) {
                var metadata = data.metadata || {};

                // Remove metadata before passing to Leaflet
                delete data.metadata;

                featureLayer = L.geoJSON(data, {
                    style: styleFeature,
                    onEachFeature: onEachFeature,
                }).addTo(map);

                // Fit bounds
                var bounds = featureLayer.getBounds();
                if (bounds.isValid()) {
                    map.fitBounds(bounds, { padding: [40, 40] });
                }

                showInfo(metadata);
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
