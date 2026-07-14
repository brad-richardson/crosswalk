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
    // Stitching-review selection-visibility colors (features 1 & 2):
    var CHANGE_COLOR = "#7C4DFF";   // glow on segments whose state differs from load
    var REMOVED_COLOR = "#9E9E9E";  // faint style for deselected (removed) segments
    var GAP_COLOR = "#FFB300";      // hazard amber for uncovered (gap) portions

    // Build an ["in", _id, [...]] expression (empty list matches nothing).
    function idIn(ids) {
        return ["in", ["get", "_id"], ["literal", ids || []]];
    }

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
    var GAP_SOURCE = "group-gaps";
    var GAP_LAYER = "group-coverage-gaps";
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

    // Per-segment visibility tracking for stitching review.
    // hiddenSegments: id -> true when the segment is NOT part of the current
    // selection ("active" === !hidden). A deselected group member stays on the
    // map in a faint "removed" style; only true context segments (never in the
    // group) are fully hidden until the user opts them in.
    var hiddenSegments = {};
    // Snapshot of the active state at group load, so we can flag CHANGES.
    var initialActiveSegments = {};   // id -> true if active when the group loaded
    var contextSegmentIds = {};       // id -> true for spatial-context segments
    var groupMemberIds = {};          // id -> true for real group segments (non-context)
    var allSegmentIds = [];           // every segment id present in the group geojson

    // Roles that carry a full segment geometry (used by removed/glow overlays).
    var FULL_ROLE_FILTER = ["in", ["get", "_role"], ["literal", ["ref-full", "target-full"]]];

    // Apply active/removed/changed styling from the current selection state.
    // - active group segments   -> normal solid layers
    // - deselected group members -> faint "removed" overlay (not hidden)
    // - context segments off     -> fully hidden
    // - segments whose active-state differs from load -> a persistent glow
    function updateSegmentStyles() {
        var activeIds = [], removedIds = [], changedIds = [];
        for (var i = 0; i < allSegmentIds.length; i++) {
            var id = allSegmentIds[i];
            var isActive = !hiddenSegments[id];
            if (isActive) {
                activeIds.push(id);
            } else if (groupMemberIds[id]) {
                removedIds.push(id); // deselected group member -> removed style
            }
            if (isActive !== !!initialActiveSegments[id]) changedIds.push(id);
        }

        // Normal role layers + labels show only ACTIVE segments.
        var normalLayers = GROUP_LAYER_DEFS.concat(GROUP_LABEL_DEFS);
        for (var j = 0; j < normalLayers.length; j++) {
            var def = normalLayers[j];
            if (!map.getLayer(def.id)) continue;
            if (def.id.indexOf("envelope") !== -1) continue; // envelope always shown
            map.setFilter(def.id, ["all", def.filter, idIn(activeIds)]);
        }

        // Removed overlay: deselected group members, faint.
        if (map.getLayer("group-ref-removed")) {
            map.setFilter("group-ref-removed",
                ["all", ["==", ["get", "_role"], "ref-full"], idIn(removedIds)]);
        }
        if (map.getLayer("group-target-removed")) {
            map.setFilter("group-target-removed",
                ["all", ["==", ["get", "_role"], "target-full"], idIn(removedIds)]);
        }

        // Change glow: any segment whose active-state differs from load.
        if (map.getLayer("group-changed-glow")) {
            map.setFilter("group-changed-glow", ["all", FULL_ROLE_FILTER, idIn(changedIds)]);
        }
        if (changedIds.length > 0) flashChangedGlow();
    }

    // Backwards-compatible alias (older call sites / console helpers).
    function updateSegmentFilters() { updateSegmentStyles(); }

    // Briefly pulse the change-glow opacity so a fresh change is noticeable,
    // then settle to a persistent subtle glow. Fully guarded/defensive.
    var _flashTimers = [];
    function flashChangedGlow() {
        if (!map.getLayer("group-changed-glow")) return;
        for (var t = 0; t < _flashTimers.length; t++) clearTimeout(_flashTimers[t]);
        _flashTimers = [];
        function setOp(op) {
            try {
                if (map.getLayer("group-changed-glow")) {
                    map.setPaintProperty("group-changed-glow", "line-opacity", op);
                }
            } catch (e) {}
        }
        setOp(0.9);
        var steps = [[120, 0.4], [260, 0.75], [420, 0.5]];
        for (var s = 0; s < steps.length; s++) {
            (function (step) {
                _flashTimers.push(setTimeout(function () { setOp(step[1]); }, step[0]));
            })(steps[s]);
        }
    }

    function toggleSegment(segmentId) {
        hiddenSegments[segmentId] = !hiddenSegments[segmentId];
        updateSegmentStyles();
        return !hiddenSegments[segmentId]; // return visible state
    }

    // Set a segment to an explicit visibility (used by the option picker to
    // pre-seed / apply an assignment without relying on prior toggle state).
    function setSegmentVisible(segmentId, visible) {
        hiddenSegments[segmentId] = !visible;
        updateSegmentStyles();
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
        updateSegmentStyles();
        return !anyVisible; // return new visible state
    }

    // Bulk-set visibility for an explicit list of segment ids in a SINGLE
    // restyle pass. Used by the per-side "select all" / "clear all" controls,
    // which pass the group pill ids for one side. Deterministic (unlike
    // toggleAllSegments, which flips) so repeated all-on/all-off flashing to
    // gauge a group's extent is smooth and predictable. Returns `visible`.
    function setSegmentsVisible(ids, visible) {
        if (ids) {
            for (var i = 0; i < ids.length; i++) {
                hiddenSegments[ids[i]] = !visible;
            }
        }
        updateSegmentStyles();
        return visible;
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

        // Feature 1 — change glow + removed overlays. Inserted BENEATH the solid
        // ref-full layer (glow first so it sits lowest, then the removed style).
        var beforeFull = map.getLayer("group-ref-full") ? "group-ref-full" : undefined;
        map.addLayer({
            id: "group-changed-glow",
            type: "line",
            source: GROUP_SOURCE,
            filter: ["all", FULL_ROLE_FILTER, idIn([])],
            paint: { "line-color": CHANGE_COLOR, "line-width": 9, "line-opacity": 0.5, "line-blur": 3 },
        }, beforeFull);
        map.addLayer({
            id: "group-ref-removed",
            type: "line",
            source: GROUP_SOURCE,
            filter: ["all", ["==", ["get", "_role"], "ref-full"], idIn([])],
            paint: { "line-color": REMOVED_COLOR, "line-width": 1.5, "line-opacity": 0.35, "line-dasharray": [1, 3] },
        }, beforeFull);
        map.addLayer({
            id: "group-target-removed",
            type: "line",
            source: GROUP_SOURCE,
            filter: ["all", ["==", ["get", "_role"], "target-full"], idIn([])],
            paint: { "line-color": REMOVED_COLOR, "line-width": 1.5, "line-opacity": 0.35, "line-dasharray": [1, 3] },
        }, beforeFull);

        // Feature 2 — coverage-gap overlay (separate source; dashed hazard amber).
        if (!map.getSource(GAP_SOURCE)) {
            map.addSource(GAP_SOURCE, { type: "geojson", data: EMPTY_FC });
        }
        map.addLayer({
            id: GAP_LAYER,
            type: "line",
            source: GAP_SOURCE,
            paint: {
                "line-color": GAP_COLOR,
                "line-width": 4,
                "line-opacity": 0.95,
                "line-dasharray": [1.5, 1.5],
            },
        });

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
        var extraLayers = [GAP_LAYER, "group-changed-glow", "group-ref-removed", "group-target-removed"];
        for (var e = 0; e < extraLayers.length; e++) {
            if (map.getLayer(extraLayers[e])) map.removeLayer(extraLayers[e]);
        }
        for (var i = 0; i < GROUP_LAYER_DEFS.length; i++) {
            if (map.getLayer(GROUP_LAYER_DEFS[i].id)) {
                map.removeLayer(GROUP_LAYER_DEFS[i].id);
            }
        }
        if (map.getSource(GAP_SOURCE)) {
            map.removeSource(GAP_SOURCE);
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
        contextSegmentIds = {};

        // Initialize context segments as hidden (user must opt-in via pills)
        var ctxEl = document.getElementById("group-context-ids");
        if (ctxEl) {
            try {
                var ctxIds = JSON.parse(ctxEl.textContent);
                for (var i = 0; i < ctxIds.length; i++) {
                    hiddenSegments[ctxIds[i]] = true;
                    contextSegmentIds[ctxIds[i]] = true;
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

        // Catalogue every segment id, split into context vs real group members,
        // and snapshot the active state at load so we can flag CHANGES (feature 1).
        allSegmentIds = [];
        groupMemberIds = {};
        initialActiveSegments = {};
        var seenIds = {};
        var feats = data.features || [];
        for (var f = 0; f < feats.length; f++) {
            var props = feats[f].properties || {};
            var sid = props._id;
            if (!sid || seenIds[sid]) continue;
            seenIds[sid] = true;
            allSegmentIds.push(sid);
            if (!contextSegmentIds[sid]) groupMemberIds[sid] = true;
            initialActiveSegments[sid] = !hiddenSegments[sid];
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

        // Apply active/removed/changed styling (context segments start hidden)
        updateSegmentStyles();

        // Render coverage gaps for the initial (pre-seeded) selection. The page
        // owns the selection logic; ask it for the included edge set if present.
        if (typeof window.matcherComputeIncludedEdges === "function") {
            try {
                var initialSel = window.matcherComputeSelection
                    ? window.matcherComputeSelection()
                    : window.matcherComputeIncludedEdges();
                updateCoverageGaps(initialSel);
            } catch (e) {}
        }

        // Fit bounds with panel-aware padding
        fitCurrentGroup();
    }

    // ---- Feature 2: coverage-gap computation ----------------------------------
    // For every ACTIVE segment, render the portions of its geometry NOT covered
    // by any currently-selected edge's alignment interval as a hazard overlay.

    // Slice a LineString's coordinates between two length-fractions using planar
    // (coordinate-space) cumulative length — matching the server's shapely
    // substring so gaps line up with the rendered aligned sub-segments.
    function sliceLineByFrac(coords, startFrac, endFrac) {
        if (!coords || coords.length < 2) return null;
        var segLens = [], total = 0;
        for (var i = 1; i < coords.length; i++) {
            var dx = coords[i][0] - coords[i - 1][0];
            var dy = coords[i][1] - coords[i - 1][1];
            var d = Math.sqrt(dx * dx + dy * dy);
            segLens.push(d);
            total += d;
        }
        if (total <= 0) return null;
        var startD = startFrac * total, endD = endFrac * total;
        function pointAt(dist) {
            if (dist <= 0) return coords[0];
            if (dist >= total) return coords[coords.length - 1];
            var acc = 0;
            for (var j = 0; j < segLens.length; j++) {
                if (acc + segLens[j] >= dist) {
                    var r = segLens[j] > 0 ? (dist - acc) / segLens[j] : 0;
                    return [
                        coords[j][0] + (coords[j + 1][0] - coords[j][0]) * r,
                        coords[j][1] + (coords[j + 1][1] - coords[j][1]) * r,
                    ];
                }
                acc += segLens[j];
            }
            return coords[coords.length - 1];
        }
        var out = [pointAt(startD)];
        var accD = 0;
        for (var k = 0; k < coords.length; k++) {
            if (k > 0) accD += segLens[k - 1];
            if (accD > startD && accD < endD) out.push(coords[k]);
        }
        out.push(pointAt(endD));
        // Drop consecutive duplicate points that can occur at boundaries.
        var cleaned = [out[0]];
        for (var m = 1; m < out.length; m++) {
            var p = out[m], q = cleaned[cleaned.length - 1];
            if (p[0] !== q[0] || p[1] !== q[1]) cleaned.push(p);
        }
        return cleaned.length >= 2 ? cleaned : null;
    }

    // Merge [start,end] intervals and return the uncovered complement within
    // [0,1] whose length exceeds minGap.
    function uncoveredIntervals(intervals, minGap) {
        if (!intervals.length) return [[0, 1]];
        var sorted = intervals.slice().sort(function (a, b) { return a[0] - b[0]; });
        var merged = [sorted[0].slice()];
        for (var i = 1; i < sorted.length; i++) {
            var last = merged[merged.length - 1];
            if (sorted[i][0] <= last[1] + 1e-9) {
                last[1] = Math.max(last[1], sorted[i][1]);
            } else {
                merged.push(sorted[i].slice());
            }
        }
        var gaps = [];
        var cursor = 0;
        for (var g = 0; g < merged.length; g++) {
            if (merged[g][0] - cursor > minGap) gaps.push([cursor, merged[g][0]]);
            cursor = Math.max(cursor, merged[g][1]);
        }
        if (1 - cursor > minGap) gaps.push([cursor, 1]);
        return gaps;
    }

    var GAP_MIN_FRAC = 0.03; // ignore gaps under ~3% of segment length

    function buildCoverageGapFC(selection) {
        // selection: {includedEdges, activeRefs, activeTargets} (see
        // computeEffectiveSegments in page.html). A legacy bare edge array is
        // upgraded by deriving actives from edge endpoints.
        var features = [];
        if (Array.isArray(selection)) {
            var derivedR = {}, derivedT = {};
            selection.forEach(function (e) { derivedR[e.ref_id] = true; derivedT[e.target_id] = true; });
            selection = { includedEdges: selection, activeRefs: derivedR, activeTargets: derivedT };
        }
        selection = selection || {};
        var includedEdges = selection.includedEdges || [];
        if (!currentGroupGeojson) {
            return { type: "FeatureCollection", features: features };
        }
        // Full-geometry coord lookup per side.
        var refGeom = {}, tgtGeom = {};
        var feats = currentGroupGeojson.features || [];
        for (var i = 0; i < feats.length; i++) {
            var props = feats[i].properties || {};
            var geom = feats[i].geometry;
            if (!geom || geom.type !== "LineString" || !props._id) continue;
            if (props._role === "ref-full" && !refGeom[props._id]) refGeom[props._id] = geom.coordinates;
            if (props._role === "target-full" && !tgtGeom[props._id]) tgtGeom[props._id] = geom.coordinates;
        }
        // Collect covered intervals per active segment.
        var refIvl = {}, tgtIvl = {};
        for (var e = 0; e < includedEdges.length; e++) {
            var ed = includedEdges[e];
            var rs = ed.gers_start_frac, re = ed.gers_end_frac;
            if (rs != null && re != null && refGeom[ed.ref_id]) {
                (refIvl[ed.ref_id] = refIvl[ed.ref_id] || []).push([Math.min(rs, re), Math.max(rs, re)]);
            }
            var ls = ed.local_start_frac, le = ed.local_end_frac;
            if (ls != null && le != null && tgtGeom[ed.target_id]) {
                (tgtIvl[ed.target_id] = tgtIvl[ed.target_id] || []).push([Math.min(ls, le), Math.max(ls, le)]);
            }
        }
        function emit(ivlMap, geomMap, role, activeIds) {
            // Every ACTIVE segment gets gap analysis — a segment with zero
            // selected intervals is 100% uncovered (full-length hazard),
            // including when the reviewer has not selected any incident edge.
            Object.keys(activeIds || {}).forEach(function (id) {
                if (!geomMap[id]) return;
                var gaps = uncoveredIntervals(ivlMap[id] || [], GAP_MIN_FRAC);
                for (var q = 0; q < gaps.length; q++) {
                    var coords = sliceLineByFrac(geomMap[id], gaps[q][0], gaps[q][1]);
                    if (coords) {
                        features.push({
                            type: "Feature",
                            geometry: { type: "LineString", coordinates: coords },
                            properties: { _role: role, _id: id },
                        });
                    }
                }
            });
        }
        emit(refIvl, refGeom, "ref-gap", selection.activeRefs);
        emit(tgtIvl, tgtGeom, "target-gap", selection.activeTargets);
        return { type: "FeatureCollection", features: features };
    }

    function updateCoverageGaps(selection) {
        if (!map.getSource(GAP_SOURCE)) return;
        try {
            map.getSource(GAP_SOURCE).setData(buildCoverageGapFC(selection));
        } catch (e) {}
    }

    /**
     * Compute fitBounds padding that keeps the geometry clear of the
     * assignment panel (#group-card). On mobile the panel is a bottom sheet
     * (reserve its height as bottom padding); on desktop it sits bottom-right
     * (reserve its width as right padding). Measured at fit time so it reflects
     * the panel's current height (collapsed vs expanded).
     */
    function computeGroupFitPadding() {
        var base = 40;
        var pad = { top: base, bottom: base, left: base, right: base };
        var card = document.getElementById("group-card");
        var mapEl = document.getElementById("map");
        if (!card || !mapEl) return pad;

        var cardRect = card.getBoundingClientRect();
        // Match the CSS mobile breakpoint (max-width: 768px is inclusive)
        var isMobile = window.innerWidth <= 768;
        if (isMobile) {
            // Bottom-sheet panel: reserve its height below the geometry.
            pad.bottom = base + cardRect.height;
        } else {
            // Bottom-right card: reserve its width to the right of the geometry.
            pad.right = base + cardRect.width;
        }

        // Clamp: MapLibre throws if padding leaves no room. Keep opposite pads
        // from summing past ~85% of the map dimension.
        var mapW = mapEl.clientWidth || window.innerWidth;
        var mapH = mapEl.clientHeight || window.innerHeight;
        var maxW = mapW * 0.85;
        var maxH = mapH * 0.85;
        if (pad.left + pad.right > maxW) pad.right = Math.max(0, maxW - pad.left);
        if (pad.top + pad.bottom > maxH) pad.bottom = Math.max(0, maxH - pad.top);
        return pad;
    }

    // Re-fit the map to the current group geometry using panel-aware padding.
    // Exposed so the collapse/expand toggle can re-fit after the panel resizes.
    function fitCurrentGroup() {
        if (!currentGroupGeojson) return;
        var bbox = geojsonBounds(currentGroupGeojson);
        if (bbox) {
            map.fitBounds(bbox, { padding: computeGroupFitPadding(), animate: false });
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
        // Include the faint "removed" overlays so a deselected segment can be
        // clicked to re-add it (it's no longer drawn by the solid full layer).
        var removedLayers = ["group-ref-removed", "group-target-removed"];
        for (var r = 0; r < removedLayers.length; r++) {
            if (map.getLayer(removedLayers[r])) layerIds.push(removedLayers[r]);
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
                // Find the matching pill button by segment id and click it.
                // Uses data-seg-id (not title) so context pills — whose title
                // now carries a human "not part of this group" hint — still
                // resolve, and group pills are unaffected.
                var pill = document.querySelector('.segment-pill[data-seg-id="' + fid + '"]');
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
    window.matcherSetSegmentsVisible = setSegmentsVisible;
    window.matcherToggleAllSegments = toggleAllSegments;
    window.matcherRefitGroup = fitCurrentGroup;
    window.matcherUpdateCoverageGaps = updateCoverageGaps;
})();
