from __future__ import annotations

import hashlib
import re
from pathlib import Path

from crosswalk.agent_labeling.matching_rubric import (
    MATCH_IDENTITY_RUBRIC,
    MATCHING_RUBRIC_VERSION,
    PAIR_LABEL_RUBRIC,
    STITCH_ASSIGNMENT_RUBRIC,
)
from crosswalk.agent_labeling.runner import prepare_batch_prompt
from crosswalk.agent_labeling.stitch_evidence import build_prompt
from crosswalk.agent_labeling.stitch_provenance import build_evidence_record

ROOT = Path(__file__).parents[2]
CANONICAL_DOC = ROOT / "docs/MATCHING_MERGING_RULES.md"


def _doc_block(name: str) -> str:
    text = CANONICAL_DOC.read_text()
    match = re.search(
        rf"<!-- BEGIN {name} -->\n(.*?)\n<!-- END {name} -->",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, f"missing canonical rubric block {name}"
    return match.group(1)


def _versioned_contract() -> str:
    text = CANONICAL_DOC.read_text()
    match = re.search(
        r"<!-- BEGIN VERSIONED_MATCHING_CONTRACT -->\n(.*?)\n"
        r"<!-- END VERSIONED_MATCHING_CONTRACT -->",
        text,
        flags=re.DOTALL,
    )
    assert match is not None, "missing versioned matching-contract markers"
    return match.group(1)


def _normalized_contract_digest() -> str:
    contract = re.sub(
        r"\(version [^)]+\)",
        "(version {MATCHING_RUBRIC_VERSION})",
        _versioned_contract(),
    )
    return hashlib.sha256(contract.encode()).hexdigest()


def _stitch_prompt() -> str:
    edges = [
        {
            "edge": "R1->T1",
            "confidence": 0.95,
            "tag": "SLIVER",
            "overlap_m": 1.2,
        },
        {
            "edge": "R1->T2",
            "confidence": 0.9,
            "tag": "BORDERLINE",
            "overlap_m": 8.0,
        },
    ]
    metadata = {
        "group_id": "rubric01",
        "match_type": "1:N",
        "n_ref_segments": 1,
        "n_target_segments": 2,
        "optimizer_letter": "A",
        "structure": {},
        "options": [
            {
                "letter": "A",
                "is_optimizer": True,
                "edge_count": 2,
                "total_confidence": 1.85,
                "mean_confidence": 0.925,
                "edges": edges,
            }
        ],
        "segments": {
            "reference": [{"label": "R1", "name": "Main", "class": "primary"}],
            "target": [
                {"label": "T1", "name": "Main", "class": "primary"},
                {"label": "T2", "name": "Main", "class": "primary"},
            ],
        },
    }
    return build_prompt(
        ROOT / "tmp-rubric-pack",
        metadata,
        {"options": [{"letter": "A"}]},
    )


def test_canonical_doc_blocks_exactly_match_runtime_rubrics():
    assert _doc_block("MATCH_IDENTITY_RUBRIC") == MATCH_IDENTITY_RUBRIC
    assert _doc_block("PAIR_LABEL_RUBRIC") == PAIR_LABEL_RUBRIC
    assert _doc_block("STITCH_ASSIGNMENT_RUBRIC") == STITCH_ASSIGNMENT_RUBRIC


def test_runtime_rubric_stamp_matches_every_embedded_version():
    for rubric in (MATCH_IDENTITY_RUBRIC, PAIR_LABEL_RUBRIC, STITCH_ASSIGNMENT_RUBRIC):
        versions = re.findall(r"\(version ([^)]+)\)", rubric)
        assert versions == [MATCHING_RUBRIC_VERSION]


def test_rubric_version_is_content_addressed_to_full_matching_contract():
    version_date, separator, version_digest = MATCHING_RUBRIC_VERSION.partition("+")
    assert separator == "+"
    assert re.fullmatch(r"\d{4}-\d{2}-\d{2}", version_date)
    assert version_digest == _normalized_contract_digest()[:12]


def test_runtime_rubrics_expose_stable_rule_ids():
    assert re.findall(r"\bMI-\d+\b", MATCH_IDENTITY_RUBRIC) == [
        "MI-1",
        "MI-2",
        "MI-3",
        "MI-4",
        "MI-5",
        "MI-6",
    ]
    assert re.findall(r"\bPL-\d+\b", PAIR_LABEL_RUBRIC) == [
        "PL-1",
        "PL-2",
        "PL-3",
        "PL-4",
    ]
    assert re.findall(r"\bSA-\d+\b", STITCH_ASSIGNMENT_RUBRIC) == [
        "SA-1",
        "SA-2",
        "SA-3",
        "SA-4",
        "SA-5",
        "SA-6",
    ]


def test_pair_prompt_embeds_only_shared_pair_contract_once():
    prompt = prepare_batch_prompt(Path("batch"), "subline_geometry_only", [], [], "out.csv")
    assert prompt.count(MATCH_IDENTITY_RUBRIC) == 1
    assert prompt.count(PAIR_LABEL_RUBRIC) == 1
    assert STITCH_ASSIGNMENT_RUBRIC not in prompt
    assert "ML FEATURE REFERENCE" not in prompt
    assert "the correct corresponding feature is a different candidate" not in prompt


def test_stitch_prompt_embeds_exact_contracts_without_old_biases():
    prompt = _stitch_prompt()
    assert prompt.count(MATCH_IDENTITY_RUBRIC) == 1
    assert prompt.count(STITCH_ASSIGNMENT_RUBRIC) == 1
    assert PAIR_LABEL_RUBRIC not in prompt
    assert "best represent" not in prompt
    assert "almost never a correct edge" not in prompt
    assert "small overlap means the segments only touch" not in prompt
    assert "finally longer overlap" not in prompt
    assert "No single signal is a universal ordering" in prompt
    assert "SLIVER(low-span/low-absolute-overlap warning)" in prompt
    assert "BORDERLINE(low span fraction, display-only)" in prompt


def test_current_evidence_records_bind_matching_rubric_version():
    group = {
        "group_id": "rubric01",
        "edges": [{"ref_id": "r1", "target_id": "t1", "confidence": 0.9}],
    }
    options = {
        "optimizer_letter": "A",
        "options": [
            {
                "letter": "A",
                "is_optimizer": True,
                "edges": [{"ref_id": "r1", "target_id": "t1", "confidence": 0.9}],
            }
        ],
    }
    evidence = build_evidence_record(group, options)
    assert evidence["matching_rubric_version"] == MATCHING_RUBRIC_VERSION
