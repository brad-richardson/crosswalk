/**
 * Persistent MapLibre GL map for the matcher web UI.
 *
 * Creates a fullscreen map centered on Boston with vector tile basemaps,
 * dark mode auto-switching, a layer group for pair geometries, and HTMX
 * integration for dynamic content loading.
 */
(function () {
    "use strict";

    // --- Basemap styles ---
    var prefersDark = window.matchMedia("(prefers-color-scheme: dark)").matches;

    var STYLES = {
        Carto: {
            light: "https://basemaps.cartocdn.com/gl/positron-gl-style/style.json",
            dark: "https://basemaps.cartocdn.com/gl/dark-matter-gl-style/style.json",
        },
        OSM: {
            // OpenMapTiles style fetched and patched to use OSM US tile server
            light: "https://openmaptiles.github.io/osm-bright-gl-style/style-cdn.json",
            dark: "https://openmaptiles.github.io/osm-bright-gl-style/style-cdn.json",
        },
        Satellite: {
            light: {
                version: 8,
                glyphs: "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf",
                sources: {
                    esri: {
                        type: "raster",
                        tiles: [
                            "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}",
                        ],
                        tileSize: 256,
                        maxzoom: 19,
                        attribution: "&copy; Esri",
                    },
                },
                layers: [{ id: "esri-satellite", type: "raster", source: "esri" }],
            },
        },
    };
    // Satellite has no dark variant
    STYLES.Satellite.dark = STYLES.Satellite.light;

    // Cache for patched OSM styles (avoid re-fetching on every switch)
    var osmStyleCache = {};

    var activeBasemap = "Carto";

    function getStyleUrl(name) {
        var entry = STYLES[name] || STYLES.Carto;
        return prefersDark ? entry.dark : entry.light;
    }

    /**
     * Patch an OpenMapTiles style to use OSM US vector tile server
     * instead of the default MapTiler API (which requires a key).
     */
    function patchOsmStyle(style) {
        var patched = JSON.parse(JSON.stringify(style));
        // Replace the openmaptiles source with OSM US tiles
        if (patched.sources && patched.sources.openmaptiles) {
            patched.sources.openmaptiles = {
                type: "vector",
                url: "https://tiles.openstreetmap.us/vector/openmaptiles.json",
            };
        }
        // Replace MapTiler font glyphs with a free alternative
        patched.glyphs = "https://demotiles.maplibre.org/font/{fontstack}/{range}.pbf";
        return patched;
    }

    /**
     * Load a style, patching OSM styles to use free tile sources.
     * Returns a promise that resolves to the style object or URL string.
     */
    function loadStyle(name) {
        var styleUrl = getStyleUrl(name);

        // Non-OSM styles: return URL directly (or inline object for Satellite)
        if (name !== "OSM") {
            return Promise.resolve(styleUrl);
        }

        // Check cache
        var cacheKey = styleUrl;
        if (osmStyleCache[cacheKey]) {
            return Promise.resolve(osmStyleCache[cacheKey]);
        }

        // Fetch and patch OSM style
        return fetch(styleUrl)
            .then(function (resp) {
                if (!resp.ok) throw new Error("Failed to load OSM style: " + resp.status);
                return resp.json();
            })
            .then(function (style) {
                var patched = patchOsmStyle(style);
                osmStyleCache[cacheKey] = patched;
                return patched;
            })
            .catch(function (err) {
                console.error("OSM style load failed, falling back to Carto:", err);
                return getStyleUrl("Carto");
            });
    }

    // --- Map initialization ---
    var map = new maplibregl.Map({
        container: "map",
        style: getStyleUrl(activeBasemap),
        center: [-71.06, 42.36],
        zoom: 14,
        maxZoom: 22,
    });

    // Add zoom control to bottom-left
    map.addControl(
        new maplibregl.NavigationControl({ showCompass: false }),
        "bottom-left"
    );

    // Add scale bar centered at top via CSS on the control corner container
    var scaleControl = new maplibregl.ScaleControl({ maxWidth: 150, unit: "metric" });
    map.addControl(scaleControl, "top-left");
    var mapContainer = map.getContainer ? map.getContainer() : document.getElementById("map");
    if (mapContainer) {
        var scaleEl = mapContainer.querySelector(".maplibregl-ctrl-top-left .maplibregl-ctrl-scale");
        if (scaleEl && scaleEl.parentElement) {
            var cornerContainer = scaleEl.parentElement;
            cornerContainer.style.display = "flex";
            cornerContainer.style.justifyContent = "center";
            cornerContainer.style.width = "100%";
            cornerContainer.style.left = "0";
            cornerContainer.style.right = "0";
            scaleEl.style.marginTop = "10px";
        }
    }

    // --- Custom layer switcher control (includes context toggle) ---
    var LayerSwitcher = (function () {
        function LayerSwitcher() {}
        LayerSwitcher.prototype.onAdd = function (map) {
            this._map = map;
            this._container = document.createElement("div");
            this._container.className = "maplibregl-ctrl maplibregl-ctrl-group layer-switcher";

            // Context toggle button (top of group)
            var ctxBtn = document.createElement("button");
            ctxBtn.type = "button";
            ctxBtn.textContent = "Context";
            ctxBtn.className = "context-toggle-btn" + (contextVisible ? " active" : "");
            ctxBtn.title = "Toggle target dataset context (C)";
            ctxBtn.addEventListener("click", toggleContextLayer);
            this._container.appendChild(ctxBtn);

            // Basemap buttons
            var names = Object.keys(STYLES);
            for (var i = 0; i < names.length; i++) {
                var btn = document.createElement("button");
                btn.type = "button";
                btn.textContent = names[i];
                btn.className = "layer-switcher-btn" + (names[i] === activeBasemap ? " active" : "");
                btn.dataset.basemap = names[i];
                btn.addEventListener("click", this._onClick.bind(this));
                this._container.appendChild(btn);
            }
            return this._container;
        };
        LayerSwitcher.prototype.onRemove = function () {
            this._container.parentNode.removeChild(this._container);
            this._map = undefined;
        };
        LayerSwitcher.prototype._onClick = function (e) {
            var name = e.target.dataset.basemap;
            if (name === activeBasemap) return;
            activeBasemap = name;

            // Update active button state
            var btns = this._container.querySelectorAll(".layer-switcher-btn");
            for (var i = 0; i < btns.length; i++) {
                btns[i].classList.toggle("active", btns[i].dataset.basemap === name);
            }

            loadStyle(name).then(function (style) {
                map.setStyle(style);
            });
        };
        return LayerSwitcher;
    })();

    map.addControl(new LayerSwitcher(), "bottom-left");

    // --- Dark mode auto-switching ---
    window.matchMedia("(prefers-color-scheme: dark)").addEventListener("change", function (e) {
        prefersDark = e.matches;
        loadStyle(activeBasemap).then(function (style) {
            map.setStyle(style);
        });
    });

    var EMPTY_FC = { type: "FeatureCollection", features: [] };

    // --- Context layer (target dataset via MVT vector tiles) ---
    var CONTEXT_SOURCE = "context-tiles";
    var CONTEXT_LAYER = "context-features-line";
    var contextDataset = null;   // dataset ID for source URL tracking
    var contextVisible = true;   // default on

    function addContextSource(dataset) {
        // Remove existing source/layer if dataset changed
        if (map.getLayer(CONTEXT_LAYER)) {
            map.removeLayer(CONTEXT_LAYER);
        }
        if (map.getSource(CONTEXT_SOURCE)) {
            map.removeSource(CONTEXT_SOURCE);
        }

        map.addSource(CONTEXT_SOURCE, {
            type: "vector",
            tiles: [window.location.origin + "/context/tiles/" + dataset + "/{z}/{x}/{y}.pbf"],
            minzoom: 10,
            maxzoom: 16,
        });

        map.addLayer({
            id: CONTEXT_LAYER,
            type: "line",
            source: CONTEXT_SOURCE,
            "source-layer": "context",
            paint: {
                "line-color": "#E57373",
                "line-width": 1.5,
                "line-opacity": 0.35,
                "line-dasharray": [2, 4],
            },
            layout: {
                visibility: contextVisible ? "visible" : "none",
            },
        });

        contextDataset = dataset;
    }

    function showContextOnMap(dataset) {
        if (!dataset) return;
        if (!map.isStyleLoaded()) return;

        // Re-add source if dataset changed or source is missing
        if (contextDataset !== dataset || !map.getSource(CONTEXT_SOURCE)) {
            addContextSource(dataset);
        }

        // Ensure context layer renders below pair layers
        if (map.getLayer(CONTEXT_LAYER) && map.getLayer("pair-reference-full")) {
            map.moveLayer(CONTEXT_LAYER, "pair-reference-full");
        }
    }

    function loadContextLayer(dataset) {
        if (!dataset) return;
        if (contextDataset === dataset && map.getSource(CONTEXT_SOURCE)) {
            // Already loaded for this dataset, just ensure visibility
            if (contextVisible && map.isStyleLoaded()) {
                showContextOnMap(dataset);
            }
            return;
        }
        if (map.isStyleLoaded()) {
            showContextOnMap(dataset);
        }
    }

    function toggleContextLayer() {
        contextVisible = !contextVisible;
        // Update map layer visibility if it exists
        if (map.getLayer(CONTEXT_LAYER)) {
            map.setLayoutProperty(
                CONTEXT_LAYER,
                "visibility",
                contextVisible ? "visible" : "none"
            );
        }
        // If turning on and source not yet added, trigger load
        if (contextVisible && !map.getSource(CONTEXT_SOURCE)) {
            var params = new URLSearchParams(window.location.search);
            var dataset = params.get("dataset");
            if (dataset) loadContextLayer(dataset);
        }
        // Update button active state
        var btn = document.querySelector(".context-toggle-btn");
        if (btn) btn.classList.toggle("active", contextVisible);
    }

    // (Context toggle is integrated into LayerSwitcher above)

    // --- Shared layer colors ---
    var REF_COLOR = "#2196F3";
    var TARGET_COLOR = "#FF5722";

    // --- Pair geometry rendering ---
    var currentGeojson = null;
    var PAIR_SOURCE = "pair-geojson";

    var LAYER_DEFS = [
        { id: "pair-reference-full", type: "line", filter: ["==", ["get", "_role"], "referenceFull"], paint: { "line-color": REF_COLOR, "line-width": 2, "line-opacity": 0.3 } },
        { id: "pair-target-full", type: "line", filter: ["==", ["get", "_role"], "targetFull"], paint: { "line-color": TARGET_COLOR, "line-width": 2, "line-opacity": 0.3, "line-dasharray": [8, 6] } },
        { id: "pair-reference", type: "line", filter: ["==", ["get", "_role"], "reference"], paint: { "line-color": REF_COLOR, "line-width": 4, "line-opacity": 0.9 } },
        { id: "pair-target", type: "line", filter: ["==", ["get", "_role"], "target"], paint: { "line-color": TARGET_COLOR, "line-width": 4, "line-opacity": 0.9 } },
    ];

    function addPairLayers() {
        if (map.getSource(PAIR_SOURCE)) return;

        map.addSource(PAIR_SOURCE, { type: "geojson", data: EMPTY_FC });

        for (var i = 0; i < LAYER_DEFS.length; i++) {
            var def = LAYER_DEFS[i];
            map.addLayer({
                id: def.id,
                type: def.type,
                source: PAIR_SOURCE,
                filter: def.filter,
                paint: def.paint,
            });
        }
    }

    /**
     * Build a FeatureCollection from pair geometry data, tagging each feature
     * with a _role property for style filtering.
     */
    function buildPairFC(data) {
        var features = [];

        function addFeatures(geojson, role) {
            if (!geojson) return;
            // Handle both Feature and FeatureCollection
            var items = geojson.type === "FeatureCollection" ? geojson.features : [geojson];
            for (var i = 0; i < items.length; i++) {
                var f = items[i];
                // If it's just a geometry (not a Feature), wrap it
                if (!f.type || f.type !== "Feature") {
                    f = { type: "Feature", geometry: f, properties: {} };
                }
                // Clone properties and add role
                var props = {};
                if (f.properties) {
                    var keys = Object.keys(f.properties);
                    for (var k = 0; k < keys.length; k++) {
                        props[keys[k]] = f.properties[keys[k]];
                    }
                }
                props._role = role;
                features.push({ type: "Feature", geometry: f.geometry, properties: props });
            }
        }

        addFeatures(data.reference_full, "referenceFull");
        addFeatures(data.target_full, "targetFull");
        addFeatures(data.reference, "reference");
        addFeatures(data.target, "target");

        return { type: "FeatureCollection", features: features };
    }

    /**
     * Compute a [west, south, east, north] bounding box from a GeoJSON object.
     */
    function geojsonBounds(geojson) {
        var coords = [];

        function extractCoords(geometry) {
            if (!geometry) return;
            if (geometry.type === "Feature") { extractCoords(geometry.geometry); return; }
            if (geometry.type === "FeatureCollection") {
                for (var i = 0; i < geometry.features.length; i++) extractCoords(geometry.features[i]);
                return;
            }
            if (geometry.type === "GeometryCollection") {
                for (var j = 0; j < geometry.geometries.length; j++) extractCoords(geometry.geometries[j]);
                return;
            }
            flattenCoords(geometry.coordinates);
        }

        function flattenCoords(c) {
            if (typeof c[0] === "number") { coords.push(c); return; }
            for (var i = 0; i < c.length; i++) flattenCoords(c[i]);
        }

        extractCoords(geojson);
        if (coords.length === 0) return null;

        var w = coords[0][0], s = coords[0][1], e = coords[0][0], n = coords[0][1];
        for (var i = 1; i < coords.length; i++) {
            if (coords[i][0] < w) w = coords[i][0];
            if (coords[i][0] > e) e = coords[i][0];
            if (coords[i][1] < s) s = coords[i][1];
            if (coords[i][1] > n) n = coords[i][1];
        }
        return [[w, s], [e, n]];
    }

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
        if (!geojsonData) {
            currentGeojson = null;
            if (map.getSource(PAIR_SOURCE)) {
                map.getSource(PAIR_SOURCE).setData(EMPTY_FC);
            }
            return;
        }

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

        currentGeojson = data;

        // Defer rendering if style isn't loaded yet
        if (!map.isStyleLoaded()) return;
        renderOverlays(data);
    }

    function renderOverlays(data) {
        if (!data) return;

        // Ensure context layer exists below pair layers
        if (contextVisible && contextDataset) {
            showContextOnMap(contextDataset);
        }

        addPairLayers();

        // Re-order context below pairs if both exist
        if (map.getLayer(CONTEXT_LAYER) && map.getLayer("pair-reference-full")) {
            map.moveLayer(CONTEXT_LAYER, "pair-reference-full");
        }

        var fc = buildPairFC(data);
        map.getSource(PAIR_SOURCE).setData(fc);

        // Fit bounds to primary geometries (reference + target sublines)
        var boundsFC = { type: "FeatureCollection", features: [] };
        if (data.reference) {
            var ref = data.reference;
            if (ref.type === "FeatureCollection") {
                boundsFC.features = boundsFC.features.concat(ref.features);
            } else {
                boundsFC.features.push(ref);
            }
        }
        if (data.target) {
            var tgt = data.target;
            if (tgt.type === "FeatureCollection") {
                boundsFC.features = boundsFC.features.concat(tgt.features);
            } else {
                boundsFC.features.push(tgt);
            }
        }

        var bbox = geojsonBounds(boundsFC);
        if (bbox) {
            var isMobile = window.innerWidth < 768;
            map.fitBounds(bbox, { padding: isMobile ? 150 : 60, animate: false });
        }
    }

    // Re-add overlays after style change (setStyle strips all custom sources/layers)
    map.on("style.load", function () {
        // Re-create source and layers
        if (map.getSource(PAIR_SOURCE)) return; // already present (initial load)
        // Re-add context layer first (so it's below pair layers)
        if (contextVisible && contextDataset) {
            // Source was stripped by style change, force re-add
            var ds = contextDataset;
            contextDataset = null;
            showContextOnMap(ds);
        }
        if (currentGeojson) {
            renderOverlays(currentGeojson);
        }
    });

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
    map.on("load", function () {
        loadPairGeometry();
        // Render any group geometry that arrived before the map was ready
        if (currentGroupGeojson) {
            renderGroupOverlays(currentGroupGeojson);
        }
        // Load context layer for current dataset
        var params = new URLSearchParams(window.location.search);
        var dataset = params.get("dataset");
        if (dataset) loadContextLayer(dataset);
    });

    // --- HTMX integration ---
    // After HTMX swaps in new content, re-read the geometry data
    document.addEventListener("htmx:afterSwap", function () {
        loadPairGeometry();
    });

    // --- Group geometry rendering (stitching review) ---
    var GROUP_SOURCE = "group-geojson";
    var currentGroupGeojson = null;

    var GROUP_LAYER_DEFS = [
        // Tier 0: Envelope polygon (very faint background)
        { id: "group-envelope", type: "fill", filter: ["==", ["get", "_role"], "envelope"], paint: { "fill-color": "#ffffff", "fill-opacity": 0.05 } },
        { id: "group-envelope-border", type: "line", filter: ["==", ["get", "_role"], "envelope"], paint: { "line-color": "#888", "line-width": 1, "line-opacity": 0.4, "line-dasharray": [6, 4] } },
        // Tier 1: Full geometries — solid lines, moderate opacity
        { id: "group-ref-full", type: "line", filter: ["==", ["get", "_role"], "ref-full"], paint: { "line-color": REF_COLOR, "line-width": 2, "line-opacity": 0.7 } },
        { id: "group-target-full", type: "line", filter: ["==", ["get", "_role"], "target-full"], paint: { "line-color": TARGET_COLOR, "line-width": 2, "line-opacity": 0.7 } },
        // Tier 2: Aligned sub-segments (thick, bright) — matches labeling pair layer colors
        { id: "group-ref-aligned", type: "line", filter: ["==", ["get", "_role"], "ref-aligned"], paint: { "line-color": REF_COLOR, "line-width": 4, "line-opacity": 0.9 } },
        { id: "group-target-aligned", type: "line", filter: ["==", ["get", "_role"], "target-aligned"], paint: { "line-color": TARGET_COLOR, "line-width": 4, "line-opacity": 0.9 } },
    ];

    // Label layers — always visible, not affected by segment toggle filters
    var GROUP_LABEL_DEFS = [
        { id: "group-ref-labels", filter: ["==", ["get", "_role"], "ref-full"], color: REF_COLOR },
        { id: "group-target-labels", filter: ["==", ["get", "_role"], "target-full"], color: TARGET_COLOR },
    ];

    // Per-segment visibility tracking for stitching review
    // Keys are segment IDs, values are booleans (true = hidden)
    var hiddenSegments = {};

    function updateSegmentFilters() {
        var hiddenIds = Object.keys(hiddenSegments).filter(function(id) { return hiddenSegments[id]; });
        var allLayers = GROUP_LAYER_DEFS.concat(GROUP_LABEL_DEFS);
        for (var i = 0; i < allLayers.length; i++) {
            var def = allLayers[i];
            var layerId = def.id;
            if (!map.getLayer(layerId)) continue;
            var roleFilter = def.filter;
            if (hiddenIds.length > 0) {
                map.setFilter(layerId, ["all", roleFilter, ["!", ["in", ["get", "_id"], ["literal", hiddenIds]]]]);
            } else {
                map.setFilter(layerId, roleFilter);
            }
        }
    }

    function toggleSegment(segmentId) {
        hiddenSegments[segmentId] = !hiddenSegments[segmentId];
        updateSegmentFilters();
        return !hiddenSegments[segmentId]; // return visible state
    }

    // Set a segment to an explicit visibility (used by the option picker to
    // pre-seed / apply an assignment without relying on prior toggle state).
    function setSegmentVisible(segmentId, visible) {
        hiddenSegments[segmentId] = !visible;
        updateSegmentFilters();
        return visible;
    }

    function toggleAllSegments(side) {
        // Determine current state: if any on this side are visible, hide all; otherwise show all
        var features = currentGroupGeojson ? currentGroupGeojson.features || [] : [];
        var sideIds = [];
        for (var i = 0; i < features.length; i++) {
            var role = features[i].properties._role || "";
            var id = features[i].properties._id;
            if (!id) continue;
            if (side === "ref" && role.indexOf("ref") === 0 && sideIds.indexOf(id) === -1) sideIds.push(id);
            if (side === "target" && role.indexOf("target") === 0 && sideIds.indexOf(id) === -1) sideIds.push(id);
        }
        // If any are visible, hide all; otherwise show all
        var anyVisible = false;
        for (var j = 0; j < sideIds.length; j++) {
            if (!hiddenSegments[sideIds[j]]) { anyVisible = true; break; }
        }
        for (var k = 0; k < sideIds.length; k++) {
            hiddenSegments[sideIds[k]] = anyVisible; // hide if any were visible
        }
        updateSegmentFilters();
        return !anyVisible; // return new visible state
    }

    function addGroupLayers() {
        if (map.getSource(GROUP_SOURCE)) return;

        map.addSource(GROUP_SOURCE, { type: "geojson", data: EMPTY_FC });

        for (var i = 0; i < GROUP_LAYER_DEFS.length; i++) {
            var def = GROUP_LAYER_DEFS[i];
            map.addLayer({
                id: def.id,
                type: def.type,
                source: GROUP_SOURCE,
                filter: def.filter,
                paint: def.paint,
            });
        }

        // Add always-visible label layers on top
        for (var j = 0; j < GROUP_LABEL_DEFS.length; j++) {
            var ldef = GROUP_LABEL_DEFS[j];
            map.addLayer({
                id: ldef.id,
                type: "symbol",
                source: GROUP_SOURCE,
                filter: ldef.filter,
                layout: {
                    "symbol-placement": "line-center",
                    "text-field": ["get", "_label"],
                    "text-size": 13,
                    "text-font": ["Open Sans Bold", "Arial Unicode MS Bold"],
                    "text-allow-overlap": true,
                    "text-ignore-placement": true,
                },
                paint: {
                    "text-color": ldef.color,
                    "text-halo-color": "#000",
                    "text-halo-width": 2,
                },
            });
        }
    }

    function removeGroupLayers() {
        for (var i = 0; i < GROUP_LABEL_DEFS.length; i++) {
            if (map.getLayer(GROUP_LABEL_DEFS[i].id)) {
                map.removeLayer(GROUP_LABEL_DEFS[i].id);
            }
        }
        for (var i = 0; i < GROUP_LAYER_DEFS.length; i++) {
            if (map.getLayer(GROUP_LAYER_DEFS[i].id)) {
                map.removeLayer(GROUP_LAYER_DEFS[i].id);
            }
        }
        if (map.getSource(GROUP_SOURCE)) {
            map.removeSource(GROUP_SOURCE);
        }
    }

    /**
     * Display group geometries on the map for stitching review.
     *
     * Expected: GeoJSON FeatureCollection with two tiers:
     * - _role "ref-full"/"target-full": full segment geometries (thin, faded)
     * - _role "ref-aligned"/"target-aligned": aligned sub-segments (thick, bright)
     */
    function showGroupGeometry(geojsonData) {
        if (!geojsonData) {
            currentGroupGeojson = null;
            if (map.getSource(GROUP_SOURCE)) {
                map.getSource(GROUP_SOURCE).setData(EMPTY_FC);
            }
            return;
        }

        var data;
        if (typeof geojsonData === "string") {
            try {
                data = JSON.parse(geojsonData);
            } catch (e) {
                console.error("Failed to parse group geometry:", e);
                return;
            }
        } else {
            data = geojsonData;
        }

        currentGroupGeojson = data;
        hiddenSegments = {}; // Reset per-segment visibility for new group

        // Initialize context segments as hidden (user must opt-in via pills)
        var ctxEl = document.getElementById("group-context-ids");
        if (ctxEl) {
            try {
                var ctxIds = JSON.parse(ctxEl.textContent);
                for (var i = 0; i < ctxIds.length; i++) {
                    hiddenSegments[ctxIds[i]] = true;
                }
            } catch (e) {}
        }

        // Initialize group segments the optimizer left out as hidden, so the
        // map matches the pre-seeded pill state (verify, don't construct).
        var inactiveEl = document.getElementById("group-inactive-ids");
        if (inactiveEl) {
            try {
                var inactiveIds = JSON.parse(inactiveEl.textContent);
                for (var m = 0; m < inactiveIds.length; m++) {
                    hiddenSegments[inactiveIds[m]] = true;
                }
            } catch (e) {}
        }

        if (!map.isStyleLoaded()) return;
        renderGroupOverlays(data);
    }

    function renderGroupOverlays(data) {
        if (!data) return;

        // Remove pair layers if present (different mode)
        if (map.getSource(PAIR_SOURCE)) {
            map.getSource(PAIR_SOURCE).setData(EMPTY_FC);
        }

        // Ensure context layer exists below group layers
        if (contextVisible && contextDataset) {
            showContextOnMap(contextDataset);
        }

        addGroupLayers();

        // Re-order context below groups
        if (map.getLayer(CONTEXT_LAYER) && map.getLayer("group-ref-full")) {
            map.moveLayer(CONTEXT_LAYER, "group-ref-full");
        }

        map.getSource(GROUP_SOURCE).setData(data);

        // Apply hidden segment filters (context segments start hidden)
        updateSegmentFilters();

        // Fit bounds
        var bbox = geojsonBounds(data);
        if (bbox) {
            var isMobile = window.innerWidth < 768;
            map.fitBounds(bbox, { padding: isMobile ? 150 : 60, animate: false });
        }
    }

    // Re-add group overlays after style change
    map.on("style.load", function () {
        if (currentGroupGeojson) {
            // Remove stale source first (style change strips them)
            if (!map.getSource(GROUP_SOURCE)) {
                renderGroupOverlays(currentGroupGeojson);
            }
        }
    });

    // --- Click-to-toggle segments on map (stitching review) ---
    map.on("click", function (e) {
        if (!currentGroupGeojson) return;

        // Query features in a bbox around click point
        var radius = 10;
        var bbox = [
            [e.point.x - radius, e.point.y - radius],
            [e.point.x + radius, e.point.y + radius],
        ];

        // Query group geometry layers + label layers (not envelope)
        var layerIds = [];
        for (var i = 0; i < GROUP_LAYER_DEFS.length; i++) {
            var lid = GROUP_LAYER_DEFS[i].id;
            if (map.getLayer(lid) && lid.indexOf("envelope") === -1) {
                layerIds.push(lid);
            }
        }
        for (var k = 0; k < GROUP_LABEL_DEFS.length; k++) {
            var llid = GROUP_LABEL_DEFS[k].id;
            if (map.getLayer(llid)) layerIds.push(llid);
        }
        if (layerIds.length === 0) return;

        var features = map.queryRenderedFeatures(bbox, { layers: layerIds });
        if (features.length === 0) return;

        // Collect unique segment IDs from hits
        var seen = {};
        for (var j = 0; j < features.length; j++) {
            var fid = features[j].properties._id;
            if (fid && !seen[fid]) {
                seen[fid] = true;
                // Find the matching pill button by title attribute and click it
                var pill = document.querySelector('.segment-pill[title="' + fid + '"]');
                if (pill) {
                    pill.click();
                }
            }
        }
    });

    // Expose map and helpers for console debugging and other scripts
    window.matcherMap = map;
    window.matcherPairLayer = PAIR_SOURCE;
    window.matcherShowGeometry = showPairGeometry;
    window.matcherShowGroupGeometry = showGroupGeometry;
    window.matcherGeojsonBounds = geojsonBounds;
    window.matcherToggleContext = toggleContextLayer;
    window.matcherToggleSegment = toggleSegment;
    window.matcherSetSegmentVisible = setSegmentVisible;
    window.matcherToggleAllSegments = toggleAllSegments;
})();
