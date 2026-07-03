"""Unit tests for stitching review modules.

Tests alternatives generation, batch selection, stitching label store,
compute_group_id, and stitching label data integrity.
"""

import json

import pandas as pd
import pytest

from matcher.labeling.stitching_store import DEFAULT_STITCHING_DIR, STITCHING_LABEL_COLUMNS
from matcher.matching.alternatives import generate_top_k_alternatives
from matcher.matching.batch_selection import select_stitching_batch
from matcher.matching.optimizer import compute_group_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _edge(ref, tgt, conf):
    return {"ref_id": ref, "target_id": tgt, "confidence": conf}


@pytest.fixture
def edges_1_to_n():
    """1:N component — one ref matched to three targets."""
    return [
        _edge("r1", "t1", 0.9),
        _edge("r1", "t2", 0.8),
        _edge("r1", "t3", 0.3),
    ]


@pytest.fixture
def edges_m_to_n():
    """M:N component — two refs matched to two targets."""
    return [
        _edge("r1", "t1", 0.9),
        _edge("r1", "t2", 0.4),
        _edge("r2", "t1", 0.3),
        _edge("r2", "t2", 0.85),
    ]


def _make_group(group_id, edges, alternatives=None, match_type="N:1"):
    """Build a minimal group dict for batch selection tests."""
    return {
        "group_id": group_id,
        "match_type": match_type,
        "edges": edges,
        "alternatives": alternatives or [],
    }


# ---------------------------------------------------------------------------
# compute_group_id
# ---------------------------------------------------------------------------


class TestComputeGroupId:
    def test_deterministic(self):
        a = compute_group_id({"r1", "r2"}, {"t1"})
        b = compute_group_id({"r2", "r1"}, {"t1"})
        assert a == b

    def test_length_is_8(self):
        assert len(compute_group_id({"r1"}, {"t1", "t2"})) == 8

    def test_different_ids_differ(self):
        a = compute_group_id({"r1"}, {"t1"})
        b = compute_group_id({"r1"}, {"t2"})
        assert a != b

    def test_ref_target_order_matters(self):
        """Swapping ref and target sets should produce a different ID."""
        a = compute_group_id({"r1"}, {"t1"})
        b = compute_group_id({"t1"}, {"r1"})
        assert a != b


# ---------------------------------------------------------------------------
# generate_top_k_alternatives
# ---------------------------------------------------------------------------


class TestGenerateTopKAlternatives:
    def test_empty_input(self):
        assert generate_top_k_alternatives([]) == []

    def test_1_to_n_returns_sorted_by_confidence(self, edges_1_to_n):
        alts = generate_top_k_alternatives(edges_1_to_n, k=5)
        confidences = [a["total_confidence"] for a in alts]
        assert confidences == sorted(confidences, reverse=True)

    def test_1_to_n_top_alternative_is_greedy(self, edges_1_to_n):
        alts = generate_top_k_alternatives(edges_1_to_n, k=5)
        top = alts[0]
        # Greedy picks all three targets assigned to r1
        edge_pairs = {(e["ref_id"], e["target_id"]) for e in top["edges"]}
        assert ("r1", "t1") in edge_pairs
        assert ("r1", "t2") in edge_pairs
        assert ("r1", "t3") in edge_pairs

    def test_m_to_n_has_multiple_alternatives(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=5)
        assert len(alts) > 1

    def test_option_index_sequential(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=5)
        indices = [a["option_index"] for a in alts]
        assert indices == list(range(len(alts)))

    def test_k_limits_results(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=2)
        assert len(alts) <= 2

    def test_no_duplicate_edge_sets(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=10)
        edge_sets = []
        for a in alts:
            es = frozenset((e["ref_id"], e["target_id"]) for e in a["edges"])
            edge_sets.append(es)
        assert len(edge_sets) == len(set(edge_sets))

    def test_summary_present(self, edges_m_to_n):
        alts = generate_top_k_alternatives(edges_m_to_n, k=1)
        assert "summary" in alts[0]
        assert "->" in alts[0]["summary"]

    @pytest.mark.parametrize(
        "edges",
        [
            [_edge("r1", "t1", 0.5)],
            [_edge("r1", "t1", 0.5), _edge("r1", "t2", 0.3)],
        ],
        ids=["single_edge", "two_edges"],
    )
    def test_always_returns_at_least_one(self, edges):
        alts = generate_top_k_alternatives(edges, k=5)
        assert len(alts) >= 1

    def test_n_to_1_enumerates_ref_subsets(self):
        """N:1 group should enumerate which refs map to the single target."""
        edges = [
            _edge("r1", "t1", 0.9),
            _edge("r2", "t1", 0.8),
            _edge("r3", "t1", 0.3),
        ]
        alts = generate_top_k_alternatives(edges, k=10)
        # 3 refs -> 2^3 - 1 = 7 non-empty subsets
        assert len(alts) == 7
        # Top alternative should include all three refs (highest total)
        top = alts[0]
        assert len(top["edges"]) == 3
        assert top["total_confidence"] == pytest.approx(2.0, abs=0.01)

    def test_duplicate_edges_keeps_highest_confidence(self):
        edges = [
            _edge("r1", "t1", 0.5),
            _edge("r1", "t1", 0.9),
        ]
        alts = generate_top_k_alternatives(edges, k=5)
        # Only one unique edge set possible
        assert len(alts) == 1
        assert alts[0]["edges"][0]["confidence"] == 0.9

    def test_alignment_fracs_preserved(self):
        """Alignment fractions from input edges should flow through to output."""
        edges = [
            {
                "ref_id": "r1",
                "target_id": "t1",
                "confidence": 0.9,
                "gers_start_frac": 0.0,
                "gers_end_frac": 0.5,
                "local_start_frac": 0.0,
                "local_end_frac": 1.0,
            },
            {
                "ref_id": "r1",
                "target_id": "t2",
                "confidence": 0.8,
                "gers_start_frac": 0.5,
                "gers_end_frac": 1.0,
                "local_start_frac": 0.0,
                "local_end_frac": 1.0,
            },
        ]
        alts = generate_top_k_alternatives(edges, k=3)
        top = alts[0]
        # Top alternative has both edges; check fracs are preserved
        for edge in top["edges"]:
            assert "gers_start_frac" in edge
            assert "gers_end_frac" in edge
            assert "local_start_frac" in edge
            assert "local_end_frac" in edge
        # Verify specific values
        t1_edge = next(e for e in top["edges"] if e["target_id"] == "t1")
        assert t1_edge["gers_start_frac"] == 0.0
        assert t1_edge["gers_end_frac"] == 0.5

    def test_alignment_fracs_absent_when_not_in_input(self):
        """Edges without alignment fracs should not have them in output."""
        edges = [_edge("r1", "t1", 0.9)]
        alts = generate_top_k_alternatives(edges, k=1)
        assert "gers_start_frac" not in alts[0]["edges"][0]


# ---------------------------------------------------------------------------
# Contiguous multi-ref chains (the "T4 spans R3+R5" gap, panel eval 2026-07-03)
# ---------------------------------------------------------------------------


def _line(coords):
    """GeoJSON-style LineString mapping (as stored in the groups sidecar)."""
    return {"type": "LineString", "coordinates": coords}


def _targets_to_refs(alt):
    """Map each target -> frozenset of refs it is assigned to in an alternative."""
    by_t: dict[str, set[str]] = {}
    for e in alt["edges"]:
        by_t.setdefault(e["target_id"], set()).add(e["ref_id"])
    return {t: frozenset(rs) for t, rs in by_t.items()}


class TestMultiRefContiguousChains:
    """A target that spans multiple CONTIGUOUS refs must be an expressible option."""

    def _spanning_group(self, contiguous: bool):
        """t1 has edges to two refs (ra, rb); rc serves t2. ra/rb contiguous iff asked."""
        edges = [
            _edge("ra", "t1", 0.9),
            _edge("rb", "t1", 0.8),
            _edge("rc", "t2", 0.85),
        ]
        # Shared exact endpoint [0.001, 0] makes ra/rb contiguous (distance 0,
        # robust to the lon/lat->meter reprojection inside the generator).
        rb_coords = [[0.001, 0], [0.002, 0]] if contiguous else [[5, 5], [5.001, 5]]
        ref_geoms = {
            "ra": _line([[0, 0], [0.001, 0]]),
            "rb": _line(rb_coords),
            "rc": _line([[9, 9], [9.001, 9]]),
        }
        return edges, ref_geoms

    def test_multiref_chain_option_generated_and_ranks(self):
        """The {ra->t1, rb->t1, ...} span is generated AND ranks into the top-K."""
        edges, ref_geoms = self._spanning_group(contiguous=True)
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=5)
        # Some option assigns t1 to BOTH ra and rb (the previously inexpressible shape).
        assert any(_targets_to_refs(a).get("t1") == frozenset({"ra", "rb"}) for a in alts)
        # It should rank at/near the top: the max-confidence option keeps both
        # ra->t1 and rb->t1 (0.9 + 0.8) alongside rc->t2.
        top = alts[0]
        assert _targets_to_refs(top).get("t1") == frozenset({"ra", "rb"})

    def test_chain_requires_contiguity(self):
        """Two refs that both touch t1 but are NOT contiguous form no chain."""
        edges, ref_geoms = self._spanning_group(contiguous=False)
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=10)
        assert not any(_targets_to_refs(a).get("t1") == frozenset({"ra", "rb"}) for a in alts)

    def test_chain_is_subset_of_existing_edges(self):
        """A chain is only offered when EVERY constituent edge exists in the group."""
        # rb is contiguous to ra but only has an edge to t2, not t1.
        edges = [
            _edge("ra", "t1", 0.9),
            _edge("rb", "t2", 0.8),
        ]
        ref_geoms = {
            "ra": _line([[0, 0], [0.001, 0]]),
            "rb": _line([[0.001, 0], [0.002, 0]]),
        }
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=10)
        # t1 never pairs with rb: the (rb, t1) edge does not exist.
        assert not any("rb" in _targets_to_refs(a).get("t1", frozenset()) for a in alts)

    def test_no_geoms_means_no_multiref(self):
        """Without geometry, behaviour is unchanged: only single-ref options."""
        edges, _ = self._spanning_group(contiguous=True)
        alts = generate_top_k_alternatives(edges, k=10)
        for a in alts:
            for refs in _targets_to_refs(a).values():
                assert len(refs) == 1

    def test_chain_length_bounded(self):
        """Chains never exceed MAX_REF_CHAIN_LEN refs for a single target."""
        from matcher.matching.alternatives import MAX_REF_CHAIN_LEN

        # Five refs in a contiguous line, all matched to t1.
        edges = [_edge(f"r{i}", "t1", 0.9 - i * 0.05) for i in range(5)]
        # Add a second target so this stays on the M:N (per-target) path.
        edges.append(_edge("r0", "t2", 0.5))
        ref_geoms = {f"r{i}": _line([[i * 0.001, 0], [(i + 1) * 0.001, 0]]) for i in range(5)}
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=50)
        for a in alts:
            for refs in _targets_to_refs(a).values():
                assert len(refs) <= MAX_REF_CHAIN_LEN

    def test_dedup_with_chains(self):
        """Chain-derived options are still deduplicated by edge set."""
        edges, ref_geoms = self._spanning_group(contiguous=True)
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=50)
        keys = [frozenset((e["ref_id"], e["target_id"]) for e in a["edges"]) for a in alts]
        assert len(keys) == len(set(keys))

    def test_chain_enumeration_beam_bounded_on_dense_graph(self):
        """A dense adjacency graph (all refs mutually contiguous) cannot blow up
        intermediate chain enumeration: the beam caps each frontier."""
        from matcher.matching.alternatives import (
            MAX_CHAIN_FRONTIER,
            _enumerate_contiguous_chains,
        )

        n = 40  # complete graph: ~C(40,3)=9880 size-3 connected subsets exist
        refs = [f"r{i}" for i in range(n)]
        adjacency = {r: set(refs) - {r} for r in refs}
        conf = {r: 1.0 - i * 0.01 for i, r in enumerate(refs)}

        chains = _enumerate_contiguous_chains(refs, adjacency, 3, conf=conf)

        # Bounded: at most max_frontier per size step (sizes 2 and 3)
        assert len(chains) <= 2 * MAX_CHAIN_FRONTIER
        # Beam keeps the best: the top-confidence pair survives
        assert frozenset({"r0", "r1"}) in chains

    def test_large_group_bounded_and_proposes_multiref(self):
        """Big groups fall back to greedy but can still propose multi-ref edges."""
        # 8 targets (> MAX_EXHAUSTIVE_TARGETS) each fed by 3 contiguous refs ->
        # greedy path. Must stay bounded (<= k) and still surface a multi-ref span.
        edges = []
        ref_geoms = {}
        for t in range(8):
            for j in range(3):
                rid = f"r{t}_{j}"
                edges.append(_edge(rid, f"t{t}", 0.9 - j * 0.1))
                base = t * 10 + j
                ref_geoms[rid] = _line([[base * 0.001, 0], [(base + 1) * 0.001, 0]])
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=5)
        assert 0 < len(alts) <= 5
        assert any(any(len(refs) >= 2 for refs in _targets_to_refs(a).values()) for a in alts)

    def test_n_to_1_ignores_geoms_full_powerset(self):
        """N:1 mirror: refs already enumerate the full power set; geoms are a no-op."""
        edges = [
            _edge("r1", "t1", 0.9),
            _edge("r2", "t1", 0.8),
            _edge("r3", "t1", 0.3),
        ]
        ref_geoms = {
            "r1": _line([[0, 0], [0.001, 0]]),
            "r2": _line([[0.001, 0], [0.002, 0]]),
            "r3": _line([[0.002, 0], [0.003, 0]]),
        }
        alts = generate_top_k_alternatives(edges, ref_geoms=ref_geoms, k=10)
        assert len(alts) == 7  # 2^3 - 1 non-empty subsets, unchanged by geoms


# ---------------------------------------------------------------------------
# select_stitching_batch
# ---------------------------------------------------------------------------


class TestSelectStitchingBatch:
    def _alt(self, conf):
        return {"total_confidence": conf, "edges": [], "summary": ""}

    def test_empty_groups(self):
        result = select_stitching_batch([], set(), k=10)
        assert result == []

    def test_skips_already_reviewed(self):
        groups = [
            _make_group("g1", [_edge("r1", "t1", 0.9)], [self._alt(0.9)]),
            _make_group("g2", [_edge("r2", "t2", 0.8)], [self._alt(0.8)]),
        ]
        result = select_stitching_batch(groups, {"g1"}, k=10)
        ids = {g["group_id"] for g in result}
        assert "g1" not in ids
        assert "g2" in ids

    def test_all_reviewed_returns_empty(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, {"g1"}, k=10)
        assert result == []

    def test_tier_balancing_with_enough_groups(self):
        """With 30 groups including large ones, k=20 should produce all four tiers."""
        groups = []
        for i in range(30):
            conf = 0.5 + i * 0.01
            # Give some groups 10+ edges so the large tier can fill
            n_edges = 12 if i < 10 else 1
            edges = [_edge(f"r{i}_{j}", f"t{i}_{j}", conf) for j in range(n_edges)]
            groups.append(
                _make_group(
                    f"g{i}",
                    edges,
                    [self._alt(conf), self._alt(conf - 0.01)],
                )
            )
        result = select_stitching_batch(groups, set(), k=20)
        assert len(result) == 20

        tiers = {g["review_tier"] for g in result}
        assert tiers == {"large", "borderline", "low_confidence", "clear_winner"}

    @pytest.mark.parametrize("k", [5, 10, 20])
    def test_respects_k(self, k):
        groups = [
            _make_group(f"g{i}", [_edge("r1", "t1", 0.5)], [self._alt(0.5)]) for i in range(50)
        ]
        result = select_stitching_batch(groups, set(), k=k)
        assert len(result) == k

    def test_largest_group_always_selected(self):
        """The single largest group should always be selected as 'large' tier."""
        big_edges = [_edge(f"r_big_{j}", f"t_big_{j}", 0.9) for j in range(15)]
        small_edges = [_edge("r_small", "t_small", 0.9)]
        groups = [
            _make_group("small", small_edges, [self._alt(0.9)]),
            _make_group("big", big_edges, [self._alt(0.9)]),
        ]
        result = select_stitching_batch(groups, set(), k=2)
        big_group = next(g for g in result if g["group_id"] == "big")
        assert big_group["review_tier"] == "large"

    def test_review_tier_and_score_present(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, set(), k=5)
        assert "review_tier" in result[0]
        assert "review_score" in result[0]

    def test_no_internal_keys_leaked(self):
        groups = [_make_group("g1", [], [self._alt(0.9)])]
        result = select_stitching_batch(groups, set(), k=5)
        for g in result:
            assert "_n_edges" not in g
            assert "_borderline_score" not in g
            assert "_low_conf_score" not in g
            assert "_review_value" not in g

    def test_low_confidence_tier(self):
        """Groups with low best-alternative confidence should get low_confidence tier."""
        # Need enough groups so all tier slots get allocated (k=10 gives 3+3+3+1)
        groups = []
        for i in range(20):
            conf = 0.9 + i * 0.005
            groups.append(
                _make_group(
                    f"filler{i}",
                    [_edge(f"r{i}", f"t{i}", conf)],
                    [
                        {"total_confidence": conf, "edges": [{"confidence": conf}], "summary": ""},
                        {
                            "total_confidence": conf - 0.01,
                            "edges": [{"confidence": conf - 0.01}],
                            "summary": "",
                        },
                    ],
                )
            )
        # Add a distinctly low-confidence group
        groups.append(
            _make_group(
                "low",
                [_edge("rlow", "tlow", 0.2)],
                [{"total_confidence": 0.2, "edges": [{"confidence": 0.2}], "summary": ""}],
            )
        )
        result = select_stitching_batch(groups, set(), k=10)
        low_group = next((g for g in result if g["group_id"] == "low"), None)
        assert low_group is not None, "Low confidence group should be selected"
        assert low_group["review_tier"] == "low_confidence"


# ---------------------------------------------------------------------------
# StitchingLabelStore
# ---------------------------------------------------------------------------


class TestStitchingLabelStore:
    @pytest.fixture
    def store(self, tmp_path):
        from matcher.labeling.stitching_store import StitchingLabelStore

        return StitchingLabelStore("test_dataset", labels_dir=tmp_path / "stitching")

    def test_empty_store(self, store):
        assert len(store.df) == 0
        assert store.get_reviewed_group_ids("test_dataset") == set()

    def test_add_and_load(self, store):
        store.add(
            group_id="abc123",
            selected_edges=[{"ref_id": "r1", "target_id": "t1"}],
            match_type="1:N",
            num_refs=1,
            num_targets=2,
            labeler="tester",
            session_id="sess1",
        )
        assert len(store.df) == 1
        assert store.df.iloc[0]["group_id"] == "abc123"
        assert store.df.iloc[0]["match_type"] == "1:N"
        assert store.df.iloc[0]["dataset_id"] == "test_dataset"

    def test_dedup_replaces_on_same_group_id(self, store):
        for i in range(3):
            store.add(
                group_id="abc123",
                selected_edges=[{"ref_id": f"r{i}", "target_id": "t1"}],
                match_type="N:1",
                num_refs=2,
                num_targets=1,
                labeler="tester",
                session_id=f"sess{i}",
            )
        assert len(store.df) == 1
        assert store.df.iloc[0]["session_id"] == "sess2"

    def test_get_reviewed_group_ids(self, store):
        store.add("g1", [], "1:N", 1, 2, "tester", "s1")
        store.add("g2", [], "N:1", 2, 1, "tester", "s2")
        reviewed = store.get_reviewed_group_ids("test_dataset")
        assert reviewed == {"g1", "g2"}

    def test_get_reviewed_filters_by_dataset(self, store):
        from matcher.labeling.stitching_store import StitchingLabelStore

        store.add("g1", [], "1:N", 1, 2, "tester", "s1")
        # Second store for a different dataset
        store2 = StitchingLabelStore("other_dataset", labels_dir=store.labels_dir)
        store2.add("g2", [], "1:N", 1, 2, "tester", "s2")
        assert store.get_reviewed_group_ids("test_dataset") == {"g1"}
        assert store2.get_reviewed_group_ids("other_dataset") == {"g2"}

    def test_selected_edges_stored_as_json(self, store):
        edges = [{"ref_id": "r1", "target_id": "t1"}, {"ref_id": "r1", "target_id": "t2"}]
        store.add("g1", edges, "1:N", 1, 2, "tester", "s1")
        raw = store.df.iloc[0]["selected_edges"]
        parsed = json.loads(raw)
        assert len(parsed) == 2
        assert parsed[0]["ref_id"] == "r1"

    def test_persistence_across_instances(self, store):
        from matcher.labeling.stitching_store import StitchingLabelStore

        store.add("g1", [], "1:N", 1, 2, "tester", "s1")

        # New instance reads from disk
        store2 = StitchingLabelStore("test_dataset", labels_dir=store.labels_dir)
        assert len(store2.df) == 1
        assert store2.df.iloc[0]["group_id"] == "g1"

    def test_atomic_backup_exists_after_save(self, store):
        store.add("g1", [], "1:N", 1, 2, "tester", "s1")
        # First save creates no backup (no prior file)
        assert store.csv_path.exists()
        # Second save creates backup
        store.add("g2", [], "N:1", 2, 1, "tester", "s2")
        backup = store.csv_path.with_suffix(".csv.bak")
        assert backup.exists()


# ---------------------------------------------------------------------------
# Stitching label data integrity
# ---------------------------------------------------------------------------

# Discover all stitching label datasets on disk
_STITCHING_DATASETS = sorted(
    p.parent.name.removeprefix("dataset=") for p in DEFAULT_STITCHING_DIR.glob("dataset=*/data.csv")
)


@pytest.mark.skipif(not _STITCHING_DATASETS, reason="no stitching labels on disk")
class TestStitchingLabelIntegrity:
    """Ensure committed stitching labels are well-formed and internally consistent."""

    @pytest.fixture(params=_STITCHING_DATASETS)
    def label_df(self, request):
        dataset = request.param
        path = DEFAULT_STITCHING_DIR / f"dataset={dataset}" / "data.csv"
        return pd.read_csv(path)

    def test_has_required_columns(self, label_df):
        missing = set(STITCHING_LABEL_COLUMNS) - set(label_df.columns)
        assert not missing, f"Missing columns: {missing}"

    def test_no_empty_rows(self, label_df):
        assert len(label_df) > 0, "Label file exists but has no rows"

    def test_group_ids_unique(self, label_df):
        dupes = label_df["group_id"].duplicated().sum()
        assert dupes == 0, f"{dupes} duplicate group_ids"

    def test_match_types_valid(self, label_df):
        valid = {"1:1", "1:N", "N:1", "M:N"}
        actual = set(label_df["match_type"].unique())
        invalid = actual - valid
        assert not invalid, f"Invalid match_type values: {invalid}"

    def test_selected_edges_parseable(self, label_df):
        for idx, row in label_df.iterrows():
            edges = json.loads(row["selected_edges"])
            assert isinstance(edges, list), f"Row {idx}: edges is not a list"
            for edge in edges:
                assert "ref_id" in edge, f"Row {idx}: edge missing ref_id"
                assert "target_id" in edge, f"Row {idx}: edge missing target_id"

    def test_num_refs_and_targets_non_negative(self, label_df):
        """Labels with edges must have positive counts; rejections (empty edges) may have 0."""
        has_edges = label_df["selected_edges"].apply(lambda x: len(json.loads(x)) > 0)
        with_edges = label_df[has_edges]
        if len(with_edges) > 0:
            assert (with_edges["num_refs"] >= 1).all(), "num_refs must be >= 1 when edges selected"
            assert (with_edges["num_targets"] >= 1).all(), (
                "num_targets must be >= 1 when edges selected"
            )
        # Rejections should have 0 refs and 0 targets
        rejections = label_df[~has_edges]
        if len(rejections) > 0:
            assert (rejections["num_refs"] == 0).all(), "num_refs must be 0 for rejections"
            assert (rejections["num_targets"] == 0).all(), "num_targets must be 0 for rejections"


# ---------------------------------------------------------------------------
# stitch-batch retains alternatives + optimizer_assignment
# ---------------------------------------------------------------------------


class TestStitchBatchRetainsOptionData:
    """The batch file must carry the data the option-picking UI depends on.

    Mirrors what the `stitch-batch` CLI does (compute alternatives, then select
    a batch) and asserts both `alternatives` and `optimizer_assignment` survive
    onto the selected groups — they must NOT be stripped anymore.
    """

    def _synthetic_sidecar_group(self, gid):
        edges = [
            _edge("r1", "t1", 0.9),
            _edge("r1", "t2", 0.4),
            _edge("r2", "t1", 0.3),
            _edge("r2", "t2", 0.85),
        ]
        return {
            "group_id": gid,
            "match_type": "M:N",
            "ref_ids": ["r1", "r2"],
            "target_ids": ["t1", "t2"],
            "edges": edges,
            # The optimizer's own proposed assignment (from the runner sidecar)
            "optimizer_assignment": [
                {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                {"ref_id": "r2", "target_id": "t2", "confidence": 0.85},
            ],
        }

    def test_alternatives_and_optimizer_assignment_retained(self):
        groups = [self._synthetic_sidecar_group(f"g{i}") for i in range(3)]
        # Replicate the CLI's compute step
        for g in groups:
            g["alternatives"] = generate_top_k_alternatives(
                component_edges=g["edges"],
                ref_geoms={},
                target_geoms={},
                k=5,
            )
        selected = select_stitching_batch(groups, set(), k=5)
        assert selected
        for g in selected:
            assert g.get("alternatives"), "alternatives must be retained on batch groups"
            assert g.get("optimizer_assignment"), (
                "optimizer_assignment must be retained on batch groups"
            )

    def test_alternatives_carry_no_geometries(self):
        """Serialized alternatives are just ID pairs + confidence (small)."""
        alts = generate_top_k_alternatives(
            component_edges=self._synthetic_sidecar_group("g")["edges"], k=5
        )
        blob = json.dumps(alts)
        assert "coordinates" not in blob
        assert "geometry" not in blob


# ---------------------------------------------------------------------------
# _build_stitch_options — option picker + optimizer pre-seed
# ---------------------------------------------------------------------------


class TestBuildStitchOptions:
    def _mn_group(self, **overrides):
        group = {
            "group_id": "g1",
            "match_type": "M:N",
            "ref_ids": ["r1", "r2"],
            "target_ids": ["t1", "t2"],
            "edges": [
                _edge("r1", "t1", 0.9),
                _edge("r1", "t2", 0.4),
                _edge("r2", "t1", 0.3),
                _edge("r2", "t2", 0.85),
            ],
            "optimizer_assignment": [
                {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                {"ref_id": "r2", "target_id": "t2", "confidence": 0.85},
            ],
            "alternatives": [],
        }
        group.update(overrides)
        return group

    def test_preseed_from_optimizer_assignment(self):
        from matcher.web.routes.stitching import _build_stitch_options

        ctx = _build_stitch_options(self._mn_group())
        assert ctx["has_preseed"] is True
        assert ctx["preseed_active_refs"] == ["r1", "r2"]
        assert ctx["preseed_active_targets"] == ["t1", "t2"]
        # First option is the optimizer's, carrying its exact edge set
        assert ctx["options"][0]["is_optimizer"] is True
        opt_edges = {(e["ref_id"], e["target_id"]) for e in ctx["options"][0]["edges"]}
        assert opt_edges == {("r1", "t1"), ("r2", "t2")}
        # preseed_edges mirrors the optimizer option so an unmodified pick
        # submits exactly that edge set
        assert ctx["preseed_edges"] == ctx["options"][0]["edges"]

    def test_missing_optimizer_assignment_falls_back(self):
        """Old-format group without optimizer_assignment -> no pre-seed."""
        from matcher.web.routes.stitching import _build_stitch_options

        group = self._mn_group()
        del group["optimizer_assignment"]
        ctx = _build_stitch_options(group)
        assert ctx["has_preseed"] is False
        assert ctx["preseed_active_refs"] is None
        assert ctx["preseed_active_targets"] is None
        assert ctx["preseed_inactive_ids"] == []
        # No optimizer option present
        assert all(not o["is_optimizer"] for o in ctx["options"])

    def test_empty_optimizer_assignment_falls_back(self):
        """An empty optimizer_assignment (optimizer dropped the group) -> no pre-seed."""
        from matcher.web.routes.stitching import _build_stitch_options

        ctx = _build_stitch_options(self._mn_group(optimizer_assignment=[]))
        assert ctx["has_preseed"] is False
        assert ctx["preseed_active_refs"] is None

    def test_inactive_segment_ids_derived(self):
        """Group segments the optimizer left out are reported inactive."""
        from matcher.web.routes.stitching import _build_stitch_options

        group = self._mn_group(
            ref_ids=["r1", "r2", "r3"],
            target_ids=["t1"],
            edges=[
                _edge("r1", "t1", 0.9),
                _edge("r2", "t1", 0.8),
                _edge("r3", "t1", 0.2),
            ],
            optimizer_assignment=[
                {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                {"ref_id": "r2", "target_id": "t1", "confidence": 0.8},
            ],
        )
        ctx = _build_stitch_options(group)
        assert ctx["preseed_active_refs"] == ["r1", "r2"]
        assert ctx["preseed_inactive_ids"] == ["r3"]

    def test_options_deduplicated_against_optimizer(self):
        """An alternative identical to the optimizer's answer is not duplicated."""
        from matcher.web.routes.stitching import _build_stitch_options

        group = self._mn_group(
            alternatives=[
                {
                    "edges": [
                        {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                        {"ref_id": "r2", "target_id": "t2", "confidence": 0.85},
                    ],
                    "total_confidence": 1.75,
                },
                {
                    "edges": [
                        {"ref_id": "r1", "target_id": "t1", "confidence": 0.9},
                    ],
                    "total_confidence": 0.9,
                },
            ]
        )
        ctx = _build_stitch_options(group)
        edge_sets = [
            frozenset((e["ref_id"], e["target_id"]) for e in o["edges"]) for o in ctx["options"]
        ]
        # The optimizer's edge set appears exactly once despite the matching alt
        assert edge_sets.count(frozenset({("r1", "t1"), ("r2", "t2")})) == 1


# ---------------------------------------------------------------------------
# _parse_explicit_edges — edge-set fidelity validation
# ---------------------------------------------------------------------------


class TestParseExplicitEdges:
    def _group(self):
        return {
            "edges": [
                _edge("r1", "t1", 0.9),
                _edge("r2", "t2", 0.85),
            ]
        }

    def test_empty_payload_returns_none(self):
        from matcher.web.routes.stitching import _parse_explicit_edges

        assert _parse_explicit_edges("", self._group()) is None

    def test_valid_payload_stripped_to_id_pairs(self):
        from matcher.web.routes.stitching import _parse_explicit_edges

        raw = json.dumps([{"ref_id": "r1", "target_id": "t1", "confidence": 0.9}])
        parsed = _parse_explicit_edges(raw, self._group())
        assert parsed == [{"ref_id": "r1", "target_id": "t1"}]

    def test_non_group_edge_rejected(self):
        from matcher.web.routes.stitching import _parse_explicit_edges

        raw = json.dumps([{"ref_id": "r1", "target_id": "t9"}])
        with pytest.raises(ValueError):
            _parse_explicit_edges(raw, self._group())

    def test_malformed_json_rejected(self):
        from matcher.web.routes.stitching import _parse_explicit_edges

        with pytest.raises(ValueError):
            _parse_explicit_edges("not json", self._group())

    def test_non_list_rejected(self):
        from matcher.web.routes.stitching import _parse_explicit_edges

        with pytest.raises(ValueError):
            _parse_explicit_edges(json.dumps({"ref_id": "r1"}), self._group())


# ---------------------------------------------------------------------------
# /stitching-review/select — explicit edge set vs. cross-product
# ---------------------------------------------------------------------------

pytest.importorskip("fastapi", reason="fastapi not installed")


class TestStitchingSelectRoute:
    """Edge-set fidelity of the submit endpoint (M:N group)."""

    DATASET = "test_ds"

    def _batch(self):
        # Full 2x2 cross-product so an option is a strict subset of the
        # cross-product of its endpoints.
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gmn",
                    "match_type": "M:N",
                    "ref_ids": ["r1", "r2"],
                    "target_ids": ["t1", "t2"],
                    "edges": [
                        _edge("r1", "t1", 0.9),
                        _edge("r1", "t2", 0.4),
                        _edge("r2", "t1", 0.3),
                        _edge("r2", "t2", 0.85),
                    ],
                }
            ],
        }

    def _client_and_recorder(self):
        from unittest.mock import MagicMock, patch

        from fastapi.testclient import TestClient

        from matcher.web.app import create_app

        recorder = MagicMock()
        patches = [
            patch("matcher.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("matcher.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch("matcher.web.routes.stitching.get_unreviewed_stitch_groups", return_value=[]),
            patch("matcher.web.routes.stitching.record_stitching_label", recorder),
        ]
        for p in patches:
            p.start()
        client = TestClient(create_app())
        return client, recorder, patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_explicit_edges_stored_verbatim(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "t1,t2",
                    "selected_edges": json.dumps(
                        [
                            {"ref_id": "r1", "target_id": "t1"},
                            {"ref_id": "r2", "target_id": "t2"},
                        ]
                    ),
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            stored = {(e["ref_id"], e["target_id"]) for e in kwargs["selected_edges"]}
            # Exactly the option's 2 edges — NOT the 4-edge cross product
            assert stored == {("r1", "t1"), ("r2", "t2")}
            assert kwargs["num_refs"] == 2
            assert kwargs["num_targets"] == 2
        finally:
            self._stop(patches)

    def test_cross_product_when_no_explicit_edges(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "t1,t2",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            stored = {(e["ref_id"], e["target_id"]) for e in kwargs["selected_edges"]}
            # Unchanged behavior: cross product of active pills = all 4 edges
            assert stored == {("r1", "t1"), ("r1", "t2"), ("r2", "t1"), ("r2", "t2")}
        finally:
            self._stop(patches)

    def test_rejects_non_group_edge(self):
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1",
                    "included_targets": "t1",
                    "selected_edges": json.dumps([{"ref_id": "r1", "target_id": "t9"}]),
                },
            )
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_manual_toggle_stores_active_pill_edges(self):
        """A manual pill edit clears selected_edges; the server must record the
        cross-product of the (non-empty) active pill fields — NOT an empty set.

        This encodes exactly what the fixed client sends after the user picks an
        option then deselects some pills: included_refs/included_targets carry
        the still-active pills, selected_edges is blank. Previously the JS wrote
        those IDs too late (in htmx:configRequest, after form serialization) so
        they arrived empty and an empty label was stored.
        """
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    # User kept both refs but only target t2 active.
                    "included_refs": "r1,r2",
                    "included_targets": "t2",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            stored = {(e["ref_id"], e["target_id"]) for e in kwargs["selected_edges"]}
            assert stored == {("r1", "t2"), ("r2", "t2")}
            assert kwargs["num_refs"] == 2
            assert kwargs["num_targets"] == 1
        finally:
            self._stop(patches)

    def test_deliberate_full_deselect_stores_empty(self):
        """Deselecting EVERYTHING (empty pill fields, blank selected_edges) is a
        legitimate reject-all and must store []."""
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "",
                    "included_targets": "",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert kwargs["selected_edges"] == []
            assert kwargs["num_refs"] == 0
            assert kwargs["num_targets"] == 0
        finally:
            self._stop(patches)

    def test_inconsistent_submission_rejected(self):
        """Active-pill fields claim segments but none resolve to a group edge —
        an inconsistent submission that must 400, not silently store []."""
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    # r1 is a real ref but the only claimed target is unknown, so
                    # the cross-product against group edges is empty.
                    "included_refs": "r1",
                    "included_targets": "t_ghost",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_refs_active_but_all_targets_deselected_rejected(self):
        """Refs still active while every target is deselected cannot form an
        edge — treated as inconsistent (not a deliberate reject-all)."""
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "",
                    "selected_edges": "",
                },
            )
            assert resp.status_code == 400
            recorder.assert_not_called()
        finally:
            self._stop(patches)

    def test_explicit_empty_list_treated_as_no_payload(self):
        """selected_edges='[]' with active pills must not bypass the guard.

        A real option always has >= 1 edge; an explicit empty list falls back
        to the manual-mode path, where non-empty pill fields resolving to zero
        edges are rejected as inconsistent.
        """
        client, recorder, patches = self._client_and_recorder()
        try:
            resp = client.post(
                "/stitching-review/select",
                data={
                    "dataset": self.DATASET,
                    "group_id": "gmn",
                    "group_index": 0,
                    "included_refs": "r1,r2",
                    "included_targets": "t1,t2",
                    "selected_edges": "[]",
                },
            )
            # Cross-product of r1,r2 x t1,t2 covers all 4 group edges -> stored
            assert resp.status_code == 200
            kwargs = recorder.call_args.kwargs
            assert len(kwargs["selected_edges"]) == 4
        finally:
            self._stop(patches)


class TestStitchingDeepLink:
    """The main page route deep-links a specific group as a FULL page."""

    DATASET = "test_ds"

    def _batch(self):
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gdeep",
                    "match_type": "1:N",
                    "ref_ids": ["r1"],
                    "target_ids": ["t1", "t2"],
                    "edges": [_edge("r1", "t1", 0.9), _edge("r1", "t2", 0.8)],
                }
            ],
        }

    def _client(self, unreviewed):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from matcher.web.app import create_app

        patches = [
            patch("matcher.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("matcher.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
            patch(
                "matcher.web.routes.stitching.get_unreviewed_stitch_groups",
                return_value=unreviewed,
            ),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_deep_link_renders_full_page_even_when_reviewed(self):
        # Group already reviewed (unreviewed list empty) — deep link still works
        client, patches = self._client(unreviewed=[])
        try:
            resp = client.get(f"/stitching-review?dataset={self.DATASET}&group_id=gdeep")
            assert resp.status_code == 200
            # Full page (styles + map container), not the bare fragment
            assert "app.css" in resp.text
            assert 'id="map"' in resp.text
            assert "gdeep" in resp.text
            # Progress reflects the group's actual batch position, not the
            # reviewed count ("Group 1 of 1", never "Group 2 of 1")
            assert "Group 1 of 1" in resp.text
        finally:
            self._stop(patches)

    def test_deep_link_unknown_group_404s_without_reflecting_id(self):
        client, patches = self._client(unreviewed=[])
        try:
            resp = client.get(
                f"/stitching-review?dataset={self.DATASET}&group_id=<script>x</script>"
            )
            assert resp.status_code == 404
            assert "<script>x</script>" not in resp.text
        finally:
            self._stop(patches)


class TestStitchingUiHooks:
    """The group card exposes the DOM hooks the client-side UX JS relies on:

    - per-pill class/name data attributes (client-side summary recompute)
    - stable summary element ids the JS targets
    - the collapse/expand toggle button
    """

    DATASET = "test_ds"

    def _batch(self):
        return {
            "dataset_id": self.DATASET,
            "groups": [
                {
                    "group_id": "gui",
                    "match_type": "1:N",
                    "ref_ids": ["r1"],
                    "target_ids": ["t1", "t2"],
                    "edges": [_edge("r1", "t1", 0.9), _edge("r1", "t2", 0.8)],
                    "ref_classes": {"r1": "residential"},
                    "ref_names": {"r1": "Main St"},
                    "target_classes": {"t1": "residential", "t2": "footway"},
                    "target_names": {"t1": "Main St", "t2": "Path"},
                }
            ],
        }

    def _client(self):
        from unittest.mock import patch

        from fastapi.testclient import TestClient

        from matcher.web.app import create_app

        patches = [
            patch("matcher.web.routes.stitching.list_datasets", return_value=[self.DATASET]),
            patch("matcher.web.routes.stitching.load_stitch_batch", return_value=self._batch()),
        ]
        for p in patches:
            p.start()
        return TestClient(create_app()), patches

    def _stop(self, patches):
        for p in patches:
            p.stop()

    def test_group_fragment_exposes_ui_hooks(self):
        client, patches = self._client()
        try:
            resp = client.get(
                f"/stitching-review/group?dataset={self.DATASET}&group_id=gui&group_index=0"
            )
            assert resp.status_code == 200
            html = resp.text
            # Collapse toggle (issue 3)
            assert 'id="panel-collapse-btn"' in html
            assert "togglePanelCollapse()" in html
            # Summary element ids the JS recompute targets (issue 1)
            assert 'id="summary-class-value"' in html
            assert 'id="summary-name-value"' in html
            # Per-pill class/name data attributes drive the recompute (issue 1)
            assert 'data-cls="residential"' in html
            assert 'data-name="Main St"' in html
            assert 'data-cls="footway"' in html
        finally:
            self._stop(patches)
