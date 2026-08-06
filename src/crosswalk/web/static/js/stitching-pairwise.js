/* Mobile-first, one-pair-at-a-time exact identity review for stitch groups. */
(function () {
    "use strict";

    var review = null;
    var advanceTimer = null;

    // Monster groups are presented in chunks of at most this many cards.
    // Chunking is presentation-only: the group identity stays single and
    // decisions accumulate across chunks via partial progress saves.
    var CHUNK_SIZE = 40;

    function parseJsonElement(id, fallback) {
        var el = document.getElementById(id);
        if (!el) return fallback;
        try { return JSON.parse(el.textContent); }
        catch (e) { return fallback; }
    }

    function pairKey(pair) {
        return pair.ref_id + "\u0000" + pair.target_id;
    }

    function shortId(value) {
        if (!value) return "";
        return value.length > 16 ? value.slice(0, 8) + "…" + value.slice(-6) : value;
    }

    function storageGet(key) {
        try { return window.localStorage.getItem(key); }
        catch (e) { return null; }
    }

    function storageSet(key, value) {
        try { window.localStorage.setItem(key, value); }
        catch (e) {}
    }

    function storageRemove(key) {
        try { window.localStorage.removeItem(key); }
        catch (e) {}
    }

    function saveDraft() {
        if (!review) return;
        storageSet(review.storageKey, JSON.stringify({
            version: 1,
            signature: review.signature,
            index: review.index,
            decisions: review.decisions
        }));
    }

    function defaultDecisions(candidates) {
        var result = {};
        for (var i = 0; i < candidates.length; i++) {
            var pair = candidates[i];
            result[pairKey(pair)] = {
                resolution: pair.default_resolution || "drop",
                identity: pair.default_identity || "unsure",
                // Server-side partial saves come back as saved cards: their
                // defaults are the recorded decision, so count them reviewed.
                reviewed: !!pair.saved
            };
        }
        return result;
    }

    function restoreDraft(candidates, storageKey, signature) {
        var result = {
            index: 0,
            hasDraft: false,
            decisions: defaultDecisions(candidates)
        };
        var raw = storageGet(storageKey);
        if (!raw) return result;
        try {
            var saved = JSON.parse(raw);
            if (saved.version !== 1 || saved.signature !== signature) return result;
            result.hasDraft = true;
            result.index = Math.max(0, Math.min(Number(saved.index) || 0, candidates.length - 1));
            for (var i = 0; i < candidates.length; i++) {
                var key = pairKey(candidates[i]);
                var decision = saved.decisions && saved.decisions[key];
                if (!decision) continue;
                if (["keep", "drop"].indexOf(decision.resolution) === -1) continue;
                if (["match", "no_match", "unsure"].indexOf(decision.identity) === -1) continue;
                if (decision.resolution === "keep" && decision.identity !== "match") continue;
                result.decisions[key] = {
                    resolution: decision.resolution,
                    identity: decision.identity,
                    reviewed: !!decision.reviewed
                };
            }
        } catch (e) {}
        return result;
    }

    function currentPair() {
        return review && review.candidates[review.index];
    }

    function currentDecision() {
        var pair = currentPair();
        return pair ? review.decisions[pairKey(pair)] : null;
    }

    function setText(id, value) {
        var el = document.getElementById(id);
        if (el) el.textContent = value == null ? "" : value;
    }

    function setActive(selector, attribute, value) {
        var buttons = document.querySelectorAll(selector);
        for (var i = 0; i < buttons.length; i++) {
            buttons[i].classList.toggle("active", buttons[i].getAttribute(attribute) === value);
        }
    }

    function reviewedCount() {
        var count = 0;
        var keys = Object.keys(review.decisions);
        for (var i = 0; i < keys.length; i++) {
            if (review.decisions[keys[i]].reviewed) count++;
        }
        return count;
    }

    function firstUnreviewedIndex() {
        for (var i = 0; i < review.candidates.length; i++) {
            if (!review.decisions[pairKey(review.candidates[i])].reviewed) return i;
        }
        return -1;
    }

    function chunkTotal() {
        return Math.max(1, Math.ceil(review.candidates.length / CHUNK_SIZE));
    }

    function chunkIndex(index) {
        return Math.floor(index / CHUNK_SIZE);
    }

    function chunkEndIndex(index) {
        return Math.min((chunkIndex(index) + 1) * CHUNK_SIZE, review.candidates.length) - 1;
    }

    function closeFeatures() {
        var drawer = document.getElementById("features-drawer");
        var backdrop = document.getElementById("features-backdrop");
        if (drawer) drawer.classList.remove("open");
        if (backdrop) backdrop.classList.add("hidden");
    }

    function showPairMap() {
        var pair = currentPair();
        if (!pair) return;
        review.overview = false;
        var button = document.getElementById("pairwise-overview-btn");
        if (button) {
            button.textContent = "Group map";
            button.setAttribute("aria-pressed", "false");
        }
        if (window.matcherShowGeometry) window.matcherShowGeometry(pair.geometry);
    }

    function toggleOverview() {
        if (!review) return;
        review.overview = !review.overview;
        var button = document.getElementById("pairwise-overview-btn");
        if (review.overview) {
            if (button) {
                button.textContent = "Pair map";
                button.setAttribute("aria-pressed", "true");
            }
            if (window.matcherShowGroupGeometry) {
                window.matcherShowGroupGeometry(review.groupGeometry);
            }
        } else {
            showPairMap();
        }
    }

    function renderPair() {
        if (!review || review.candidates.length === 0) return;
        window.clearTimeout(advanceTimer);
        var pair = currentPair();
        var decision = currentDecision();
        var card = document.getElementById("pairwise-card");
        var reviewView = document.getElementById("pairwise-review-view");
        var summary = document.getElementById("pairwise-summary");
        if (reviewView) reviewView.hidden = false;
        if (summary) summary.hidden = true;
        if (card) card.classList.remove("showing-summary");

        var progress = "Pair " + (review.index + 1) + " of " + review.candidates.length;
        if (chunkTotal() > 1) {
            progress += " · chunk " + (chunkIndex(review.index) + 1) + "/" + chunkTotal();
        }
        setText("pairwise-progress", progress);
        setText("pairwise-reviewed-count", reviewedCount() + " reviewed");
        setText("pairwise-ref-name", pair.ref_name || "unnamed");
        setText("pairwise-target-name", pair.target_name || "unnamed");
        setText("pairwise-ref-id", shortId(pair.ref_id));
        setText("pairwise-target-id", shortId(pair.target_id));
        var refId = document.getElementById("pairwise-ref-id");
        var targetId = document.getElementById("pairwise-target-id");
        if (refId) {
            refId.title = pair.ref_id;
            refId.dataset.full = pair.ref_id;
            refId.dataset.short = shortId(pair.ref_id);
        }
        if (targetId) {
            targetId.title = pair.target_id;
            targetId.dataset.full = pair.target_id;
            targetId.dataset.short = shortId(pair.target_id);
        }
        setText("pairwise-classes", (pair.ref_class || "?") + " / " + (pair.target_class || "?"));
        setText(
            "pairwise-confidence",
            typeof pair.confidence === "number" ? Math.round(pair.confidence * 100) + "%" : ""
        );

        var warnings = [];
        if (!pair.geometry_available) warnings.push("geometry unavailable");
        if (pair.is_external) warnings.push("outside group");
        if (pair.is_sliver) warnings.push("sliver");
        setText("pairwise-warning", warnings.join(" · "));
        setActive("#pairwise-identity-choices button", "data-identity", decision.identity);
        setActive("#pairwise-resolution-choices button", "data-resolution", decision.resolution);
        var prev = document.getElementById("pairwise-prev");
        if (prev) prev.disabled = review.index === 0;
        var next = document.getElementById("pairwise-next");
        if (next) next.textContent = review.index === chunkEndIndex(review.index)
            ? "Review summary" : "Accept & next";

        var body = document.getElementById("features-drawer-body");
        if (body) body.innerHTML = "";
        closeFeatures();
        showPairMap();
        if (card) card.scrollTop = 0;
    }

    function showSummary() {
        if (!review) return;
        window.clearTimeout(advanceTimer);
        var reviewView = document.getElementById("pairwise-review-view");
        var summary = document.getElementById("pairwise-summary");
        var card = document.getElementById("pairwise-card");
        if (reviewView) reviewView.hidden = true;
        if (summary) summary.hidden = false;
        if (card) card.classList.add("showing-summary");

        // Only decided cards count: with partial progress the undecided
        // defaults must not inflate the identity/keep tallies.
        var counts = { match: 0, no_match: 0, unsure: 0, keep: 0 };
        var keys = Object.keys(review.decisions);
        for (var i = 0; i < keys.length; i++) {
            var decision = review.decisions[keys[i]];
            if (!decision.reviewed) continue;
            counts[decision.identity]++;
            if (decision.resolution === "keep") counts.keep++;
        }
        var total = review.candidates.length;
        var reviewed = reviewedCount();
        setText("pairwise-summary-reviewed", reviewed + " / " + total);
        setText("pairwise-summary-matches", counts.match);
        setText("pairwise-summary-nomatches", counts.no_match);
        setText("pairwise-summary-unsure", counts.unsure);
        setText("pairwise-summary-kept", counts.keep);
        var title = reviewed === total ? "Group ready to record"
            : chunkTotal() > 1
                ? "Chunk " + (chunkIndex(review.index) + 1) + " of " + chunkTotal() + " reviewed"
                : "Progress so far";
        setText("pairwise-summary-title", title);
        setText("pairwise-review-again", reviewed === total ? "Review pairs" : "Continue reviewing");
        var save = document.getElementById("pairwise-save-progress");
        if (save) {
            save.hidden = !(reviewed > 0 && reviewed < total);
            save.textContent = "Save progress (" + reviewed + " of " + total + ")";
        }
        closeFeatures();
        if (window.matcherShowGroupGeometry) {
            window.matcherShowGroupGeometry(review.groupGeometry);
        }
    }

    function advance() {
        if (!review) return;
        saveDraft();
        // Chunk boundaries (and the true end) both land on the summary, so a
        // monster group pauses every CHUNK_SIZE cards with a save affordance.
        if (review.index >= chunkEndIndex(review.index)) {
            showSummary();
            return;
        }
        review.index++;
        saveDraft();
        renderPair();
    }

    function scheduleAdvance() {
        window.clearTimeout(advanceTimer);
        advanceTimer = window.setTimeout(advance, 220);
    }

    function chooseIdentity(value) {
        var decision = currentDecision();
        if (!decision) return;
        decision.identity = value;
        if (value !== "match") decision.resolution = "drop";
        decision.reviewed = true;
        saveDraft();
        setActive("#pairwise-identity-choices button", "data-identity", decision.identity);
        setActive("#pairwise-resolution-choices button", "data-resolution", decision.resolution);
        scheduleAdvance();
    }

    function chooseResolution(value) {
        var decision = currentDecision();
        if (!decision) return;
        decision.resolution = value;
        if (value === "keep") {
            decision.identity = "match";
            decision.reviewed = true;
        }
        saveDraft();
        setActive("#pairwise-identity-choices button", "data-identity", decision.identity);
        setActive("#pairwise-resolution-choices button", "data-resolution", decision.resolution);
        if (value === "keep") scheduleAdvance();
    }

    function previous() {
        if (!review || review.index === 0) return;
        window.clearTimeout(advanceTimer);
        review.index--;
        saveDraft();
        renderPair();
    }

    function acceptAndAdvance() {
        var decision = currentDecision();
        if (!decision) return;
        decision.reviewed = true;
        advance();
    }

    function reviewAgain() {
        if (!review) return;
        var firstUnreviewed = firstUnreviewedIndex();
        review.index = firstUnreviewed >= 0 ? firstUnreviewed : 0;
        saveDraft();
        renderPair();
    }

    function openFeatures() {
        var pair = currentPair();
        var card = document.getElementById("pairwise-card");
        var drawer = document.getElementById("features-drawer");
        var backdrop = document.getElementById("features-backdrop");
        var body = document.getElementById("features-drawer-body");
        if (!pair || !card || !drawer || !body) return;
        drawer.classList.add("open");
        if (backdrop) backdrop.classList.remove("hidden");
        body.innerHTML = '<span class="spinner"></span>';
        var query = new URLSearchParams({
            dataset: card.getAttribute("data-queue-dataset"),
            group_id: card.getAttribute("data-group-id"),
            group_dataset: card.getAttribute("data-group-dataset"),
            ref_id: pair.ref_id,
            target_id: pair.target_id
        });
        if (window.htmx) {
            window.htmx.ajax("GET", "/stitching-review/pair-features?" + query.toString(), {
                target: "#features-drawer-body",
                swap: "innerHTML"
            });
        }
    }

    function submitDispositions(partial) {
        var dispositions = [];
        var selected = [];
        for (var i = 0; i < review.candidates.length; i++) {
            var pair = review.candidates[i];
            var decision = review.decisions[pairKey(pair)];
            if (partial && !decision.reviewed) continue;
            dispositions.push({
                ref_id: pair.ref_id,
                target_id: pair.target_id,
                resolution: decision.resolution,
                identity: decision.identity
            });
            if (decision.resolution === "keep") {
                selected.push({ ref_id: pair.ref_id, target_id: pair.target_id });
            }
        }
        var selectedInput = document.getElementById("pairwise-selected-edges");
        var dispositionsInput = document.getElementById("pairwise-edge-dispositions");
        var rejectInput = document.getElementById("pairwise-confirm-reject-all");
        var partialInput = document.getElementById("pairwise-partial-save");
        if (dispositionsInput) dispositionsInput.value = JSON.stringify(dispositions);
        if (partial) {
            // Progress saves never assert resolver truth: no selected_edges
            // payload, no reject-all claim. The server stores the subset under
            // the partial scope and re-serves this group with it prefilled.
            if (selectedInput) selectedInput.value = "";
            if (rejectInput) rejectInput.value = "";
            if (partialInput) partialInput.value = "1";
        } else {
            if (selectedInput) selectedInput.value = JSON.stringify(selected);
            if (rejectInput) rejectInput.value = selected.length === 0 ? "1" : "";
            if (partialInput) partialInput.value = "";
        }
        window.__pairwisePendingDraftKey = review.storageKey;
        var form = document.getElementById("pairwise-submit-form");
        if (form) form.requestSubmit();
    }

    function saveProgress() {
        if (!review) return;
        var reviewed = reviewedCount();
        if (reviewed === 0 || reviewed === review.candidates.length) return;
        submitDispositions(true);
    }

    function recordGroup() {
        if (!review) return;
        if (reviewedCount() !== review.candidates.length) {
            reviewAgain();
            return;
        }
        submitDispositions(false);
    }

    function bindButtons() {
        document.querySelectorAll("#pairwise-identity-choices button").forEach(function (button) {
            button.addEventListener("click", function () {
                chooseIdentity(button.getAttribute("data-identity"));
            });
        });
        document.querySelectorAll("#pairwise-resolution-choices button").forEach(function (button) {
            button.addEventListener("click", function () {
                chooseResolution(button.getAttribute("data-resolution"));
            });
        });
        var bindings = [
            ["pairwise-prev", previous],
            ["pairwise-next", acceptAndAdvance],
            ["pairwise-overview-btn", toggleOverview],
            ["pairwise-features", openFeatures],
            ["pairwise-features-close", closeFeatures],
            ["pairwise-features-handle", closeFeatures],
            ["features-backdrop", closeFeatures],
            ["pairwise-review-again", reviewAgain],
            ["pairwise-save-progress", saveProgress],
            ["pairwise-record", recordGroup]
        ];
        for (var i = 0; i < bindings.length; i++) {
            var element = document.getElementById(bindings[i][0]);
            if (element) element.addEventListener("click", bindings[i][1]);
        }
    }

    function initPairwiseReview() {
        window.clearTimeout(advanceTimer);
        var card = document.getElementById("pairwise-card");
        if (!card) {
            review = null;
            return;
        }
        var candidates = parseJsonElement("pairwise-candidates", []);
        var signature = candidates.map(pairKey).join("\u0001");
        var storageKey = "crosswalk:pairwise:v1:" +
            card.getAttribute("data-group-dataset") + ":" + card.getAttribute("data-group-id");
        var restored = restoreDraft(candidates, storageKey, signature);
        review = {
            candidates: candidates,
            groupGeometry: parseJsonElement("pairwise-group-geometry", null),
            signature: signature,
            storageKey: storageKey,
            index: restored.index,
            decisions: restored.decisions,
            overview: false
        };
        bindButtons();
        if (candidates.length) {
            var firstUnreviewed = firstUnreviewedIndex();
            if (!restored.hasDraft) {
                // Resume server-saved progress at the first undecided card.
                review.index = firstUnreviewed >= 0 ? firstUnreviewed : 0;
            }
            if (!restored.hasDraft && firstUnreviewed === -1 && reviewedCount() > 0) {
                // Everything decided (e.g. saved but never completed): land on
                // the summary so "Complete group" is one tap away.
                showSummary();
            } else {
                renderPair();
            }
        } else {
            setText("pairwise-progress", "No candidate pairs");
            var next = document.getElementById("pairwise-next");
            if (next) next.disabled = true;
        }
    }

    if (document.readyState === "loading") {
        document.addEventListener("DOMContentLoaded", initPairwiseReview);
    } else {
        initPairwiseReview();
    }
    document.addEventListener("htmx:afterSwap", function (event) {
        var target = event.target;
        if (target && (target.id === "group-content" ||
            (target.closest && target.closest("#group-content")))) {
            initPairwiseReview();
        }
    });
    document.addEventListener("htmx:afterRequest", function (event) {
        if (!window.__pairwisePendingDraftKey) return;
        var xhr = event.detail && event.detail.xhr;
        if (xhr && xhr.status >= 200 && xhr.status < 300) {
            storageRemove(window.__pairwisePendingDraftKey);
            window.__pairwisePendingDraftKey = null;
        } else if (xhr && xhr.status >= 400) {
            window.alert("Save failed (" + xhr.status + "). Your pairwise draft is still saved.");
            window.__pairwisePendingDraftKey = null;
        }
    });
})();
