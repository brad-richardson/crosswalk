# Deferred stitching audit

This is a durable index of stitching-label cases intentionally excluded from
adjudication-clean resolver evaluation. It is an audit/TODO list, not corrected
ground truth and not evidence that the production optimizer is wrong. The
existing fail-closed behavior remains appropriate: contradictory current groups
stay quarantined, while the legacy-known split omissions remain provenance
findings rather than universal candidate-recall failures.

The tables were generated from `build_edge_table(...).attrs["build_audit"]`
using schema version 1. Case identities use historical label IDs and exact
segment IDs; the current eight-character group ID is only a locator and may
change when components are rebuilt.

The complete, machine-readable payload is checked in as
[`stitching_deferred_audit.json`](stitching_deferred_audit.json). It preserves
all selected-edge sets, conflicting claims, and raw omission records even
though the source group sidecars are runtime artifacts. Regenerate or verify it
with `uv run python scripts/generate_stitching_deferred_audit.py [--check]`;
the generator refuses source files whose content hashes differ from this
snapshot.

## Source snapshot

- Source commit: `9adfdefa7a5893dea207bbdd17db828013d6644f`
- Source tree: `00418f8e72b8f60409bd2ad79cf12f5b98de40ff`
- Boston groups: `data/output/us_boston_streets_groups.json`, SHA-256 `5eaeadc4ac7ad1d0d061c14c58fbdecbe501ce07144ff3686ac5a77218807f28`
- Boston labels: `labels/stitching/dataset=us_boston_streets/data.csv`, SHA-256 `16014f8b5734932ffa29fcb7ba447e397f872d1cd3a99dab4b3f604af5fbf470`
- Seattle groups: `data/output/us_seattle_sidewalks_groups.json`, SHA-256 `6310d7c4180b0d212d67eae757c57a8d4dbb290847d2c9a1017c3448d8983b00`
- Seattle labels: `labels/stitching/dataset=us_seattle_sidewalks/data.csv`, SHA-256 `c7dbd3bc8fb67ccef2ab480594b5202dea4228abf4dc00ebff9115acfdc0593b`

The `data/output` sidecars are local runtime artifacts, so their hashes are
required to interpret the current group locators below.

## Quarantined label collisions

These three Boston groups account for 130 raw emitted row occurrences and 30
contradictory current candidate keys.

| Current group | Stable historical lineage | Provenance / labeler | Raw rows | Conflicting keys |
|---|---|---|---:|---:|
| `42a76195` | `3ef22541` + `5ebfa40f` | clean / brad; clean / brad | 32 | 13 |
| `658fce64` | `701d491e` + `d8f883c1` | split / brad; clean / brad | 38 | 2 |
| `6ffc5468` | `2eb37ad1` + `6ffc5468` | split / panel_unanimous_v1; clean / brad | 60 | 15 |

The checked-in structured audit contains each label's complete sorted selected-edge set
and, for every conflicting edge, the historical label, provenance, and keep
claim. Those details are intentionally not duplicated in this concise index.

## Retained legacy-known split omissions

All 14 retained keys have split provenance: 13 Boston keys and one Seattle key.
They are human-selected edges visible in the mapped current group's legacy
`edges`/`rejected_edges` view but absent from that group's emitted candidate
universe. A split label can legitimately contain an edge owned by another
current group.

| Dataset | Current group | Reference ID | Target ID | Historical label occurrence(s) | Available name |
|---|---|---|---|---|---|
| Boston | `310b338c` | `5466d710-7428-4b8a-9931-b8c9afd749c0` | `us_boston_streets_9989_882a3066b3` | `bb93702d` | Welles Avenue (target) |
| Boston | `310b338c` | `8c62758e-cdee-4ec5-a5f0-3a7e53c25546` | `us_boston_streets_9989_882a3066b3` | `bb93702d` | Welles Avenue (target) |
| Boston | `4995e327` | `83381cb5-c064-44a9-a8f5-7fa02259fbb6` | `us_boston_streets_4130_882a30660b` | `a89e4b84` | North Margin Street (target) |
| Boston | `4995e327` | `da6921cb-eb01-4638-84a7-4106412efe5c` | `us_boston_streets_4130_882a30660b` | `a89e4b84` | North Margin Street (target) |
| Boston | `4995e327` | `f6c9b841-e180-46a7-9811-2bd6cce1243a` | `us_boston_streets_4130_882a30660b` | `a89e4b84` | North Margin Street (target) |
| Boston | `6cf0a147` | `d9d22d5e-f2c6-4e25-847a-f57f83670b94` | `us_boston_streets_20_882a306639` | `cc0c30a0` | Chestnut Street (target) |
| Boston | `6eb7078b` | `cd06bc62-c453-436a-ba1e-1a8d737d65b6` | `us_boston_streets_1373_882a306419` | `f170979a` | Bragdon Street (reference) |
| Boston | `7b7c77dc` | `b02844df-c219-4bfe-9e38-932ea6727b6f` | `us_boston_streets_10413_882a3066b5` | `058c4460` | Park Street (reference) |
| Boston | `90dc1de1` | `1b670d46-c6e3-49ad-a08a-61ea60aa0c0e` | `us_boston_streets_34_882a30644b` | `9ac35fb7`, `f452e052` | Boylston Street (reference) |
| Boston | `90dc1de1` | `b3c8bbdc-0216-4bc8-8af6-1c8fae2f1b46` | `us_boston_streets_6050_882a30644b` | `9ac35fb7`, `f452e052` | Boylston Street (reference) |
| Boston | `9466ce8e` | `cd456724-7e95-4749-855a-180e383e293a` | `us_boston_streets_1609_882a3066b5` | `77d260da` | Vassar Street (reference) |
| Boston | `ec725e24` | `8cf1ac96-aa94-42eb-bbe2-64ce5ebbca8b` | `us_boston_streets_6790_882a306633` | `04fc93e5` | Rutland Street (target) |
| Boston | `f1b99028` | `a2be6d1c-e452-4288-b094-46356f60c67f` | `us_boston_streets_5818_882a30661d` | `3074ed80` | Cross Street (target) |
| Seattle | `4a7f3a7d` | `0fada7b9-a294-4f48-bf31-fcb0aa8be06a` | `sea_sidewalk_292556_8828d542dd` | `670e939f` | — |

One additional Boston raw omission key belongs to quarantined current group
`658fce64` (`701d491e`: `726ce920-9b48-4349-87e5-f824013e807a` →
`us_boston_streets_6653_882a3064c7`) and is therefore not one of the 14 retained
keys.

## Manual unresolved follow-ups

| TODO | Status | Durable handling |
|---|---|---|
| Winthrop | Deferred / unresolved | Investigate possible human under-selection after an exact historical stitching label or segment set is pinned. Several current groups and panel-context records contain “Winthrop”; do not resolve this TODO by street-name search. |
| Public Alley | Deferred / unresolved | Investigate the conservative REVIEW-policy example after its exact stitching case is pinned. Pair label `47229b83-55ba-4329-bb44-c4ba1967c6ab` → `us_boston_streets_7933_882a306639` is a related searchable lead, but the current artifact emits it as a 1:1 MATCH, so it must not be assumed to be the intended stitching case. |

When either manual TODO is identified, record exact historical label and segment
IDs here. Do not use a current group ID alone as its durable identity.
