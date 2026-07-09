"""Tests for ``crosswalk data stitch-rekey`` (#374/#375).

Stitching labels are keyed by group_id = hash of the exact ref/target id set,
so optimizer re-grouping strands them. The rekey command classifies every
label against the current sidecar and blind-rekeys ONLY the safe 1:1 clean
bucket; everything contested (N->1 merges, 1->N splits, set-semantics,
collisions with existing rows) routes to review or is refused.

The critical guarded invariant: the store must never end up with duplicate
``group_id`` rows — ``resolver/extract.py`` keys ``human_by`` on group_id
(last-row-wins) while ``recover_labeled_groups`` maps per-row, so duplicates
silently corrupt extraction and double-count eval. The known real-data hazard
is Boston's 49-labels-into-8-merged-groups collapse (#375), which a naive
``group_id = current_id`` rewrite would apply silently.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from typer.testing import CliRunner

from crosswalk.agent_labeling.stitch_eval import recover_labeled_groups
from crosswalk.cli import app
from crosswalk.labeling.stitch_rekey import (
    REKEY_LOG_FILENAME,
    apply_clean_rekey,
    build_rekey_plan,
    read_rekey_log,
)
from crosswalk.labeling.stitching_store import StitchingLabelStore

DATASET = "test_dataset"

runner = CliRunner()


# ---------------------------------------------------------------------------
# fixtures
# ---------------------------------------------------------------------------


def _group(gid: str, edges: list[tuple[str, str]]) -> dict:
    """A current-sidecar group with selected edges."""
    return {
        "group_id": gid,
        "edges": [{"ref_id": r, "target_id": t, "selected": True} for r, t in edges],
        "ref_ids": sorted({r for r, _ in edges}),
        "target_ids": sorted({t for _, t in edges}),
        "match_type": "M:N",
    }


def _store(tmp_path) -> StitchingLabelStore:
    return StitchingLabelStore(DATASET, labels_dir=tmp_path / "stitching")


def _add_pair_label(store, gid: str, edges: list[tuple[str, str]], labeler: str = "brad"):
    store.add(
        group_id=gid,
        selected_edges=[{"ref_id": r, "target_id": t} for r, t in edges],
        match_type="M:N",
        num_refs=len({r for r, _ in edges}),
        num_targets=len({t for _, t in edges}),
        labeler=labeler,
        session_id="s1",
    )


def _add_set_label(store, gid: str, ref_ids: list[str], target_ids: list[str]):
    store.add(
        group_id=gid,
        selected_edges=[],
        match_type="M:N",
        num_refs=len(ref_ids),
        num_targets=len(target_ids),
        labeler="brad",
        session_id="s1",
        label_semantics="set",
        ref_ids=ref_ids,
        target_ids=target_ids,
    )


def _write_sidecar(tmp_path, groups: list[dict]):
    p = tmp_path / "groups.json"
    p.write_text(json.dumps({"groups": groups}))
    return p


# ---------------------------------------------------------------------------
# plan buckets
# ---------------------------------------------------------------------------


class TestPlanBuckets:
    def test_clean_1to1_drift(self, tmp_path):
        """A drifted label whose edges land in one unoccupied group is clean."""
        groups = [_group("new1", [("r1", "t1"), ("r2", "t2")])]
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1"), ("r2", "t2")])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.mapping == {"old1": "new1"}
        assert [m.labeler for m in plan.clean] == ["brad"]
        assert not plan.has_refusals
        assert plan.counts()["unchanged"] == 0

    def test_unchanged_label_not_rekeyed(self, tmp_path):
        groups = [_group("g1", [("r1", "t1")])]
        store = _store(tmp_path)
        _add_pair_label(store, "g1", [("r1", "t1")])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.unchanged == ["g1"]
        assert plan.mapping == {}

    def test_n_to_1_disjoint_scopes_is_union_case(self, tmp_path):
        """Optimizer merged two disjoint labeled sub-groups: union proposal, review."""
        merged = _group("merged", [("r1", "t1"), ("r2", "t2"), ("r3", "t3")])
        store = _store(tmp_path)
        _add_pair_label(store, "oldA", [("r1", "t1")])
        _add_pair_label(store, "oldB", [("r2", "t2"), ("r3", "t3")])

        plan = build_rekey_plan([merged], store.load(DATASET), DATASET)

        assert plan.mapping == {}  # never blind-rekeyed
        assert len(plan.merge_union) == 1
        case = plan.merge_union[0]
        assert case.new_group_id == "merged"
        assert set(case.old_group_ids) == {"oldA", "oldB"}
        assert set(case.union_edges) == {("r1", "t1"), ("r2", "t2"), ("r3", "t3")}
        assert plan.merge_conflict == []
        assert not plan.has_refusals

    def test_n_to_1_overlapping_scopes_is_conflict(self, tmp_path):
        """Two labels adjudicated the same segment: union is unsafe -> conflict."""
        merged = _group("merged", [("r1", "t1"), ("r1", "t2")])
        store = _store(tmp_path)
        # Both labels made an assertion about r1 with different edge sets: the
        # union would overrule oldA's implicit rejection of (r1, t2).
        _add_pair_label(store, "oldA", [("r1", "t1")])
        _add_pair_label(store, "oldB", [("r1", "t2")])

        plan = build_rekey_plan([merged], store.load(DATASET), DATASET)

        assert plan.mapping == {}
        assert plan.merge_union == []
        assert len(plan.merge_conflict) == 1
        assert "r1" in plan.merge_conflict[0].shared_segments

    def test_split_goes_to_review(self, tmp_path):
        """A label whose edges now span two current groups is never auto-rekeyed."""
        groups = [_group("gA", [("r1", "t1")]), _group("gB", [("r2", "t2")])]
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1"), ("r2", "t2")])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.mapping == {}
        assert len(plan.split) == 1
        s = plan.split[0]
        assert s.old_group_id == "old1"
        assert s.n_edges_total == 2
        assert s.n_edges_in_best == 1

    def test_target_already_labeled_is_refused(self, tmp_path):
        """Rekeying onto a group that already has a label row would duplicate."""
        groups = [_group("new1", [("r1", "t1"), ("r2", "t2")])]
        store = _store(tmp_path)
        _add_pair_label(store, "new1", [("r1", "t1")])  # existing row at target
        _add_pair_label(store, "old1", [("r1", "t1"), ("r2", "t2")])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.mapping == {}
        assert plan.has_refusals
        assert plan.collision_refused[0].new_group_id == "new1"
        assert plan.collision_refused[0].old_group_ids == ("old1",)

    def test_set_semantics_never_auto_applied(self, tmp_path):
        """SET labels route to review with a membership-based report (#375)."""
        groups = [_group("new1", [("r1", "t1"), ("r2", "t2")])]
        store = _store(tmp_path)
        _add_set_label(store, "oldset", ref_ids=["r1", "r2"], target_ids=["t1", "t2"])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.mapping == {}
        assert len(plan.set_review) == 1
        sc = plan.set_review[0]
        assert sc.dominant_group_id == "new1"
        assert sc.n_members_total == 4
        assert sc.n_members_in_dominant == 4
        assert sc.n_groups_spanned == 1

    def test_lost_and_empty_unrecoverable(self, tmp_path):
        groups = [_group("gX", [("r9", "t9")])]
        store = _store(tmp_path)
        _add_pair_label(store, "oldgone", [("r1", "t1")])  # edges no longer exist
        _add_pair_label(store, "oldempty", [])  # reject-all, group_id gone

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.lost == ["oldgone"]
        assert plan.empty_unrecoverable == ["oldempty"]
        assert plan.mapping == {}

    def test_empty_reject_all_with_current_gid_is_unchanged(self, tmp_path):
        groups = [_group("g1", [("r1", "t1")])]
        store = _store(tmp_path)
        _add_pair_label(store, "g1", [])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.unchanged == ["g1"]
        assert plan.empty_unrecoverable == []

    def test_boston_shape_merge_collapse_never_clean(self, tmp_path):
        """The 49->8 hazard in miniature: 3 labels -> 1 merged group must not
        produce a single 'clean' winner (a naive rewrite keeps last-write-wins)."""
        merged = _group("mega", [(f"r{i}", f"t{i}") for i in range(6)])
        store = _store(tmp_path)
        _add_pair_label(store, "oldA", [("r0", "t0"), ("r1", "t1")])
        _add_pair_label(store, "oldB", [("r2", "t2"), ("r3", "t3")])
        _add_pair_label(store, "oldC", [("r4", "t4"), ("r5", "t5")])

        plan = build_rekey_plan([merged], store.load(DATASET), DATASET)

        assert plan.mapping == {}
        assert len(plan.merge_union) == 1
        assert set(plan.merge_union[0].old_group_ids) == {"oldA", "oldB", "oldC"}

    def test_pre_existing_duplicate_rows_are_refused(self, tmp_path):
        """Duplicate group_id rows in the store (corruption) trip the refusal."""
        groups = [_group("g1", [("r1", "t1")])]
        store = _store(tmp_path)
        _add_pair_label(store, "g1", [("r1", "t1")])
        # Bypass add()'s dedupe to simulate a hand-edited/corrupted CSV.
        df = store.df
        store._df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        store.save()

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan.has_refusals
        assert any("2 label rows" in c.reason for c in plan.collision_refused)


# ---------------------------------------------------------------------------
# store-level collision guards (defense in depth: the store API itself refuses)
# ---------------------------------------------------------------------------


class TestStoreRekeyGuards:
    def test_rekey_rewrites_key_and_preserves_row(self, tmp_path):
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")], labeler="brad")
        before = store.df.iloc[0].to_dict()

        n = store.rekey_group_ids({"old1": "new1"})

        assert n == 1
        reloaded = StitchingLabelStore(DATASET, labels_dir=store.labels_dir).load(DATASET)
        assert list(reloaded["group_id"]) == ["new1"]
        after = reloaded.iloc[0].to_dict()
        for col in ("labeler", "labeled_at", "selected_edges", "match_type", "session_id"):
            assert after[col] == before[col]

    def test_duplicate_target_mapping_refused(self, tmp_path):
        """Two old ids -> one new id (the N->1 collapse) must raise, write nothing."""
        store = _store(tmp_path)
        _add_pair_label(store, "oldA", [("r1", "t1")])
        _add_pair_label(store, "oldB", [("r2", "t2")])
        csv_before = store.csv_path.read_text()

        with pytest.raises(ValueError, match="not 1:1"):
            store.rekey_group_ids({"oldA": "merged", "oldB": "merged"})

        assert store.csv_path.read_text() == csv_before

    def test_occupied_target_refused(self, tmp_path):
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])
        _add_pair_label(store, "new1", [("r2", "t2")])
        csv_before = store.csv_path.read_text()

        with pytest.raises(ValueError, match="already have a label row"):
            store.rekey_group_ids({"old1": "new1"})

        assert store.csv_path.read_text() == csv_before

    def test_missing_source_refused(self, tmp_path):
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])

        with pytest.raises(ValueError, match="not found in store"):
            store.rekey_group_ids({"ghost": "new1"})

    def test_duplicate_source_rows_refused(self, tmp_path):
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])
        df = store.df
        store._df = pd.concat([df, df.iloc[[0]]], ignore_index=True)
        store.save()

        with pytest.raises(ValueError, match="duplicate label rows"):
            store.rekey_group_ids({"old1": "new1"})

    def test_swap_chain_is_atomic(self, tmp_path):
        """A new id equal to another old id being moved away is fine (vectorized)."""
        store = _store(tmp_path)
        _add_pair_label(store, "a", [("r1", "t1")])
        _add_pair_label(store, "b", [("r2", "t2")])

        n = store.rekey_group_ids({"a": "b2", "b": "a"})  # b -> a while a -> b2

        assert n == 2
        gids = set(store.load(DATASET)["group_id"])
        assert gids == {"b2", "a"}


# ---------------------------------------------------------------------------
# apply + audit trail
# ---------------------------------------------------------------------------


class TestApplyAndAudit:
    def test_apply_only_touches_clean_bucket(self, tmp_path):
        groups = [
            _group("new1", [("r1", "t1")]),
            _group("merged", [("r2", "t2"), ("r3", "t3")]),
        ]
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])  # clean
        _add_pair_label(store, "oldA", [("r2", "t2")])  # merge member
        _add_pair_label(store, "oldB", [("r3", "t3")])  # merge member

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)
        n = apply_clean_rekey(store, plan, sidecar="groups.json")

        assert n == 1
        gids = set(store.load(DATASET)["group_id"])
        assert gids == {"new1", "oldA", "oldB"}  # merge members untouched

    def test_audit_log_written_and_survives_store_roundtrip(self, tmp_path):
        """The join table must survive _ensure_schema round-trips (it is a
        sidecar file, not a label column the schema projection would drop)."""
        groups = [_group("new1", [("r1", "t1")])]
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)
        apply_clean_rekey(store, plan, sidecar="groups.json")

        log = read_rekey_log(store.partition_path)
        assert list(log["old_group_id"]) == ["old1"]
        assert list(log["new_group_id"]) == ["new1"]
        assert list(log["labeler"]) == ["brad"]
        assert list(log["sidecar"]) == ["groups.json"]

        # Full store round-trip (load -> save -> reload): the rekeyed key and
        # the audit log both survive.
        rt = StitchingLabelStore(DATASET, labels_dir=store.labels_dir)
        rt._df = rt._load()
        rt.save()
        reloaded = StitchingLabelStore(DATASET, labels_dir=store.labels_dir).load(DATASET)
        assert list(reloaded["group_id"]) == ["new1"]
        log2 = read_rekey_log(rt.partition_path)
        pd.testing.assert_frame_equal(log, log2)

    def test_audit_log_appends_across_repeated_rekeys(self, tmp_path):
        """Repeatable rekey (#375): the lineage chain accumulates, never clobbers."""
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])

        plan1 = build_rekey_plan([_group("mid1", [("r1", "t1")])], store.load(DATASET), DATASET)
        apply_clean_rekey(store, plan1, sidecar="run1.json")
        plan2 = build_rekey_plan([_group("new1", [("r1", "t1")])], store.load(DATASET), DATASET)
        apply_clean_rekey(store, plan2, sidecar="run2.json")

        log = read_rekey_log(store.partition_path)
        assert list(zip(log["old_group_id"], log["new_group_id"])) == [
            ("old1", "mid1"),
            ("mid1", "new1"),
        ]

    def test_resolver_contract_parity_after_rekey(self, tmp_path):
        """Rekeyed labels must recover verbatim through recover_labeled_groups
        (the mapping contract resolver/extract.build_edge_table relies on),
        with no duplicate group_id rows to corrupt its human_by dict."""
        groups = [
            _group("new1", [("r1", "t1"), ("r2", "t2")]),
            _group("new2", [("r3", "t3")]),
        ]
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1"), ("r2", "t2")])
        _add_pair_label(store, "old2", [("r3", "t3")])

        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)
        assert plan.mapping == {"old1": "new1", "old2": "new2"}
        apply_clean_rekey(store, plan, sidecar="groups.json")

        rekeyed = store.load(DATASET)
        # No duplicate keys: human_by = {group_id: row} is lossless.
        assert rekeyed["group_id"].is_unique
        rec = recover_labeled_groups(groups, rekeyed)
        # Every rekeyed label now maps clean onto ITS OWN current group.
        assert sorted(rec["clean"]) == [("new1", "new1"), ("new2", "new2")]
        assert rec["split"] == [] and rec["lost"] == []
        # And the resolver's per-edge extraction sees them under the new keys.
        from crosswalk.resolver.extract import build_edge_table

        table = build_edge_table(groups, rekeyed, DATASET)
        assert set(table["group_id"]) == {"new1", "new2"}
        assert set(table["human_group_id"]) == {"new1", "new2"}
        assert table["keep"].all()

    def test_rekey_is_idempotent(self, tmp_path):
        """Re-running against the same sidecar finds nothing to do."""
        groups = [_group("new1", [("r1", "t1")])]
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])
        plan = build_rekey_plan(groups, store.load(DATASET), DATASET)
        apply_clean_rekey(store, plan, sidecar="groups.json")

        plan2 = build_rekey_plan(groups, store.load(DATASET), DATASET)

        assert plan2.mapping == {}
        assert plan2.unchanged == ["new1"]
        assert apply_clean_rekey(store, plan2, sidecar="groups.json") == 0


# ---------------------------------------------------------------------------
# CLI behavior (dry-run default, --apply, exit codes)
# ---------------------------------------------------------------------------


class TestCli:
    def _setup(self, tmp_path, groups, build_store):
        sidecar = _write_sidecar(tmp_path, groups)
        store = _store(tmp_path)
        build_store(store)
        return sidecar, store

    def _invoke(self, tmp_path, sidecar, *extra):
        return runner.invoke(
            app,
            [
                "data",
                "stitch-rekey",
                DATASET,
                "--sidecar",
                str(sidecar),
                "--labels-dir",
                str(tmp_path / "stitching"),
                *extra,
            ],
        )

    def test_dry_run_default_makes_no_writes(self, tmp_path):
        sidecar, store = self._setup(
            tmp_path,
            [_group("new1", [("r1", "t1")])],
            lambda s: _add_pair_label(s, "old1", [("r1", "t1")]),
        )
        csv_before = store.csv_path.read_text()

        result = self._invoke(tmp_path, sidecar)

        assert result.exit_code == 0, result.output
        assert "old1" in result.output and "new1" in result.output
        assert "dry-run" in result.output
        assert store.csv_path.read_text() == csv_before
        assert not (store.partition_path / REKEY_LOG_FILENAME).exists()
        assert not store.csv_path.with_suffix(".csv.bak").exists()

    def test_apply_rekeys_clean_and_writes_audit(self, tmp_path):
        sidecar, store = self._setup(
            tmp_path,
            [_group("new1", [("r1", "t1")])],
            lambda s: _add_pair_label(s, "old1", [("r1", "t1")]),
        )

        result = self._invoke(tmp_path, sidecar, "--apply")

        assert result.exit_code == 0, result.output
        reloaded = StitchingLabelStore(DATASET, labels_dir=store.labels_dir).load(DATASET)
        assert list(reloaded["group_id"]) == ["new1"]
        log = read_rekey_log(store.partition_path)
        assert list(log["new_group_id"]) == ["new1"]
        assert str(sidecar) in log["sidecar"].iloc[0]

    def test_collision_exits_nonzero_and_blocks_apply(self, tmp_path):
        def build(s):
            _add_pair_label(s, "new1", [("r1", "t1")])  # occupies the target
            _add_pair_label(s, "old1", [("r1", "t1"), ("r2", "t2")])
            _add_pair_label(s, "old2", [("r3", "t3")])  # clean, but blocked

        sidecar, store = self._setup(
            tmp_path,
            [
                _group("new1", [("r1", "t1"), ("r2", "t2")]),
                _group("new2", [("r3", "t3")]),
            ],
            build,
        )
        csv_before = store.csv_path.read_text()

        dry = self._invoke(tmp_path, sidecar)
        assert dry.exit_code == 1
        assert "REFUSED" in dry.output

        applied = self._invoke(tmp_path, sidecar, "--apply")
        assert applied.exit_code == 1
        assert "nothing was written" in " ".join(applied.output.split())
        assert store.csv_path.read_text() == csv_before
        assert not (store.partition_path / REKEY_LOG_FILENAME).exists()

    def test_allow_partial_applies_clean_despite_collision(self, tmp_path):
        def build(s):
            _add_pair_label(s, "new1", [("r1", "t1")])
            _add_pair_label(s, "old1", [("r1", "t1"), ("r2", "t2")])  # refused
            _add_pair_label(s, "old2", [("r3", "t3")])  # clean

        sidecar, store = self._setup(
            tmp_path,
            [
                _group("new1", [("r1", "t1"), ("r2", "t2")]),
                _group("new2", [("r3", "t3")]),
            ],
            build,
        )

        result = self._invoke(tmp_path, sidecar, "--apply", "--allow-partial")

        assert result.exit_code == 0, result.output
        gids = set(
            StitchingLabelStore(DATASET, labels_dir=store.labels_dir).load(DATASET)["group_id"]
        )
        assert gids == {"new1", "old1", "new2"}  # old2 -> new2; old1 untouched
        log = read_rekey_log(store.partition_path)
        assert list(log["old_group_id"]) == ["old2"]

    def test_missing_sidecar_errors(self, tmp_path):
        store = _store(tmp_path)
        _add_pair_label(store, "old1", [("r1", "t1")])

        result = self._invoke(tmp_path, tmp_path / "nope.json")

        assert result.exit_code == 1
        assert "No groups sidecar" in result.output

    def test_no_labels_exits_cleanly(self, tmp_path):
        sidecar = _write_sidecar(tmp_path, [_group("g1", [("r1", "t1")])])

        result = self._invoke(tmp_path, sidecar)

        assert result.exit_code == 0
        assert "No stitching labels" in result.output
