"""Pure schema tests for per-ballot stitching evidence delivery provenance."""

from __future__ import annotations

import copy
import json

import pytest

from crosswalk.agent_labeling.stitch_provenance import (
    DELIVERY_MODE_NATIVE_ATTACHMENT,
    DELIVERY_MODE_PROMPT_PATH,
    DELIVERY_PREFLIGHT_PASSED,
    DELIVERY_SCHEMA_VERSION,
    EvidenceProvenanceError,
    build_evidence_delivery_record,
    canonical_evidence_delivery_json,
    managed_image_descriptors,
    sha256_json,
    validate_evidence_delivery_record,
)


def _manifest() -> dict:
    # Deliberately not path-sorted: the helper must emit a deterministic set.
    return {
        "schema_version": 1,
        "evidence_pack_sha256": "f" * 64,
        "files": [
            {"path": "metadata.yaml", "bytes": 20, "sha256": "0" * 64},
            {"path": "zoom_R1_T1.png", "bytes": 404, "sha256": "4" * 64},
            {"path": "overview.png", "bytes": 101, "sha256": "1" * 64},
            {"path": "prompt.txt", "bytes": 30, "sha256": "5" * 64},
            {"path": "option_B.png", "bytes": 303, "sha256": "3" * 64},
            {"path": "option_A.png", "bytes": 202, "sha256": "2" * 64},
        ],
    }


def test_managed_image_descriptors_are_pack_relative_and_canonical():
    assets = managed_image_descriptors(_manifest())

    assert [asset["path"] for asset in assets] == [
        "option_A.png",
        "option_B.png",
        "overview.png",
        "zoom_R1_T1.png",
    ]
    assert all(set(asset) == {"path", "bytes", "sha256"} for asset in assets)
    assert not any(asset["path"].startswith("/") for asset in assets)


@pytest.mark.parametrize(
    ("delivery_mode", "transport"),
    [
        (DELIVERY_MODE_NATIVE_ATTACHMENT, "codex:-i"),
        (DELIVERY_MODE_NATIVE_ATTACHMENT, "opencode:-f"),
        (DELIVERY_MODE_PROMPT_PATH, "claude:Read"),
        (DELIVERY_MODE_PROMPT_PATH, "agy:agent-read"),
    ],
)
def test_build_and_validate_canonical_delivery_record(delivery_mode, transport):
    manifest = _manifest()
    record = build_evidence_delivery_record(
        manifest,
        delivery_mode=delivery_mode,
        transport=transport,
    )
    encoded = canonical_evidence_delivery_json(record)

    assert record == {
        "schema_version": DELIVERY_SCHEMA_VERSION,
        "delivery_mode": delivery_mode,
        "transport": transport,
        "preflight_status": DELIVERY_PREFLIGHT_PASSED,
        "asset_count": 4,
        "asset_set_sha256": sha256_json(managed_image_descriptors(manifest)),
        "assets": managed_image_descriptors(manifest),
    }
    assert encoded == json.dumps(record, sort_keys=True, separators=(",", ":"))
    assert (
        validate_evidence_delivery_record(
            encoded,
            manifest,
            expected_delivery_mode=delivery_mode,
            expected_transport=transport,
        )
        == record
    )


@pytest.mark.parametrize(
    ("delivery_mode", "transport", "match"),
    [
        (DELIVERY_MODE_NATIVE_ATTACHMENT, "claude:Read", "not valid"),
        (DELIVERY_MODE_PROMPT_PATH, "codex:-i", "not valid"),
        ("telepathy", "codex:-i", "unknown"),
        (DELIVERY_MODE_NATIVE_ATTACHMENT, 7, "must be a string"),
    ],
)
def test_build_rejects_unknown_or_incompatible_transport(delivery_mode, transport, match):
    with pytest.raises(EvidenceProvenanceError, match=match):
        build_evidence_delivery_record(
            _manifest(),
            delivery_mode=delivery_mode,
            transport=transport,
        )


def test_build_only_allows_passed_preflight():
    with pytest.raises(EvidenceProvenanceError, match="requires passed preflight"):
        build_evidence_delivery_record(
            _manifest(),
            delivery_mode=DELIVERY_MODE_NATIVE_ATTACHMENT,
            transport="codex:-i",
            preflight_status="failed",
        )


def _valid_record() -> dict:
    return build_evidence_delivery_record(
        _manifest(),
        delivery_mode=DELIVERY_MODE_NATIVE_ATTACHMENT,
        transport="codex:-i",
    )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (lambda r: r.update(schema_version=2), "unsupported.*schema"),
        (lambda r: r.update(schema_version=True), "unsupported.*schema"),
        (lambda r: r.update(delivery_mode=DELIVERY_MODE_PROMPT_PATH), "not valid"),
        (lambda r: r.update(transport="opencode:-f"), "transport mismatch"),
        (lambda r: r.update(preflight_status="failed"), "preflight did not pass"),
        (lambda r: r.update(asset_count=3), "asset_count"),
        (lambda r: r.update(asset_count=4.0), "asset_count"),
        (lambda r: r.update(asset_set_sha256="0" * 64), "asset_set_sha256"),
        (lambda r: r["assets"][0].update(sha256="a" * 64), "do not match"),
        (lambda r: r["assets"].reverse(), "do not match"),
        (lambda r: r["assets"].append(copy.deepcopy(r["assets"][0])), "do not match"),
        (lambda r: r.update(consumed=True), "fields mismatch"),
        (lambda r: r["assets"][0].update(runtime_path="/tmp/x"), "exactly"),
    ],
)
def test_validation_rejects_tampered_records(mutate, match):
    record = _valid_record()
    mutate(record)

    with pytest.raises(EvidenceProvenanceError, match=match):
        validate_evidence_delivery_record(
            record,
            _manifest(),
            expected_delivery_mode=DELIVERY_MODE_NATIVE_ATTACHMENT,
            expected_transport="codex:-i",
        )


def test_validation_rejects_noncanonical_json():
    record = _valid_record()
    noncanonical = json.dumps(record, indent=2)

    with pytest.raises(EvidenceProvenanceError, match="not canonical"):
        validate_evidence_delivery_record(noncanonical, _manifest())


def test_validation_rejects_expected_mode_or_transport_mismatch():
    record = _valid_record()

    with pytest.raises(EvidenceProvenanceError, match="mode mismatch"):
        validate_evidence_delivery_record(
            record,
            _manifest(),
            expected_delivery_mode=DELIVERY_MODE_PROMPT_PATH,
        )
    with pytest.raises(EvidenceProvenanceError, match="transport mismatch"):
        validate_evidence_delivery_record(
            record,
            _manifest(),
            expected_transport="opencode:-f",
        )


@pytest.mark.parametrize(
    ("mutate", "match"),
    [
        (
            lambda m: m["files"].append({"path": "overview.png", "bytes": 101, "sha256": "1" * 64}),
            "duplicate",
        ),
        (
            lambda m: m["files"].__setitem__(
                2, {"path": "/tmp/overview.png", "bytes": 101, "sha256": "1" * 64}
            ),
            "unsupported image",
        ),
        (
            lambda m: m["files"].__setitem__(
                2, {"path": "overview.png", "bytes": -1, "sha256": "1" * 64}
            ),
            "non-negative",
        ),
        (
            lambda m: m["files"].__setitem__(
                2, {"path": "overview.png", "bytes": 101, "sha256": "XYZ"}
            ),
            "invalid sha256",
        ),
        (
            lambda m: m["files"].__setitem__(
                2, {"path": "other.png", "bytes": 101, "sha256": "1" * 64}
            ),
            "unsupported image",
        ),
        (
            lambda m: m["files"].pop(2),
            "no overview",
        ),
    ],
)
def test_manifest_image_mismatches_fail_closed(mutate, match):
    manifest = _manifest()
    mutate(manifest)

    with pytest.raises(EvidenceProvenanceError, match=match):
        managed_image_descriptors(manifest)
