"""Unit tests for the access/mode evidence channel (Part B).

Evidence-pack only — NOT an ML feature. Covers Overture extraction
(``parse_access_lr`` + class-default pass), the never-inferred-denial guard, the
prompt ``access=`` rendering, and the ``route_network`` header.
"""

from __future__ import annotations

from crosswalk.agent_labeling.stitch_evidence import _edge_struct_str, build_prompt
from crosswalk.fetch.overture import parse_access_lr
from crosswalk.utils.physical import summarize_access


def _mode_map(lr) -> dict:
    """Collapse a single-span access LR dict-list to its per-mode map."""
    assert len(lr) == 1, lr
    assert lr[0]["between"] == [0.0, 1.0]
    return lr[0]["value"]


# ---------------------------------------------------------------------------
# (a) parse_access_lr maps the sample access_restrictions shapes
# ---------------------------------------------------------------------------


def test_parse_access_lr_tagged_allowed_and_denied() -> None:
    # Clean unconditional shape: motor_vehicle denied, foot allowed (no scoped
    # carve-out) → hard tagged values survive.
    restrictions = [
        {"access_type": "denied", "when": {"mode": ["motor_vehicle"]}},
        {"access_type": "allowed", "when": {"mode": ["foot"]}},
    ]
    modes = _mode_map(parse_access_lr(restrictions, road_class="footway").to_dict_list())
    assert modes["motor_vehicle"] == {"value": "denied", "source": "tagged"}
    assert modes["foot"] == {"value": "allowed", "source": "tagged"}
    # bicycle stays unknown (never guessed) → omitted from the stored map.
    assert "bicycle" not in modes


def test_parse_access_lr_blanket_allowed_no_scope() -> None:
    # when.mode null + no scope → unconditional allow across all modes.
    restrictions = [{"access_type": "allowed", "when": {}}]
    modes = _mode_map(parse_access_lr(restrictions, road_class="residential").to_dict_list())
    for mode in ("motor_vehicle", "bicycle", "foot"):
        assert modes[mode] == {"value": "allowed", "source": "tagged"}


def test_parse_access_lr_single_entry_multiple_modes() -> None:
    restrictions = [{"access_type": "denied", "when": {"mode": ["bicycle", "foot"]}}]
    modes = _mode_map(parse_access_lr(restrictions, road_class="motorway").to_dict_list())
    # Both listed modes take the tagged denial (here it agrees with the motorway
    # class default, but the source is tagged because the entry named them).
    assert modes["bicycle"] == {"value": "denied", "source": "tagged"}
    assert modes["foot"] == {"value": "denied", "source": "tagged"}
    # motor_vehicle untouched by the entry → class default.
    assert modes["motor_vehicle"] == {"value": "allowed", "source": "class_default"}


def test_parse_access_lr_tagged_designated() -> None:
    restrictions = [{"access_type": "designated", "when": {"mode": ["bicycle"]}}]
    # class=path so no cycleway default masks the tagged designation.
    modes = _mode_map(parse_access_lr(restrictions, road_class="path").to_dict_list())
    assert modes["bicycle"] == {"value": "designated", "source": "tagged"}


def test_parse_access_lr_recognized_scope_is_restricted() -> None:
    # when.mode null + when.recognized scope → restricted across all modes.
    restrictions = [{"access_type": "allowed", "when": {"recognized": ["as_private"]}}]
    modes = _mode_map(parse_access_lr(restrictions, road_class=None).to_dict_list())
    for mode in ("motor_vehicle", "bicycle", "foot"):
        assert modes[mode] == {"value": "restricted", "source": "tagged"}


def test_parse_access_lr_during_scope_is_restricted() -> None:
    restrictions = [
        {
            "access_type": "denied",
            "when": {"mode": ["motor_vehicle"], "during": "Mo-Fr 07:00-19:00"},
        }
    ]
    modes = _mode_map(parse_access_lr(restrictions, road_class="residential").to_dict_list())
    # Time-scoped entry is restricted, not a hard denial — and it overrides the
    # driveable class-default (tagged beats class_default).
    assert modes["motor_vehicle"] == {"value": "restricted", "source": "tagged"}


def test_parse_access_lr_heading_entries_ignored() -> None:
    # Heading-scoped entries belong to the direction/oneway channel — the access
    # channel must ignore them and fall back to class defaults only.
    restrictions = [{"access_type": "denied", "when": {"heading": "backward"}}]
    modes = _mode_map(parse_access_lr(restrictions, road_class="residential").to_dict_list())
    assert modes == {"motor_vehicle": {"value": "allowed", "source": "class_default"}}


def test_parse_access_lr_tagged_overrides_class_default() -> None:
    # A tagged motor_vehicle=denied must win over the driveable class default.
    restrictions = [{"access_type": "denied", "when": {"mode": ["motor_vehicle"]}}]
    modes = _mode_map(parse_access_lr(restrictions, road_class="residential").to_dict_list())
    assert modes["motor_vehicle"] == {"value": "denied", "source": "tagged"}


def test_parse_access_lr_denied_then_scoped_allow_exception_is_restricted() -> None:
    # Unconditional deny + a scoped allowed carve-out (private/delivery access)
    # is denied-with-exceptions → restricted, NOT a hard tagged denial — even
    # though the deny is listed FIRST (no entry-ordering artifact).
    restrictions = [
        {"access_type": "denied", "when": {"mode": ["motor_vehicle"]}},
        {
            "access_type": "allowed",
            "when": {"mode": ["motor_vehicle"], "recognized": ["as_delivery"]},
        },
    ]
    modes = _mode_map(parse_access_lr(restrictions, road_class="residential").to_dict_list())
    assert modes["motor_vehicle"] == {"value": "restricted", "source": "tagged"}


def test_parse_access_lr_blanket_private_exception_softens_denial() -> None:
    # The verified 3-entry footway shape: mv denied + foot allowed + a blanket
    # "allowed when recognized=as_private" carve-out over all modes. The private
    # carve-out softens the mv denial to restricted; the unconditional foot allow
    # dominates its own scoped mention; bicycle sees only the scoped carve-out.
    restrictions = [
        {"access_type": "denied", "when": {"mode": ["motor_vehicle"]}},
        {"access_type": "allowed", "when": {"mode": ["foot"]}},
        {"access_type": "allowed", "when": {"recognized": ["as_private"]}},
    ]
    modes = _mode_map(parse_access_lr(restrictions, road_class="footway").to_dict_list())
    assert modes["motor_vehicle"] == {"value": "restricted", "source": "tagged"}
    assert modes["foot"] == {"value": "allowed", "source": "tagged"}
    assert modes["bicycle"] == {"value": "restricted", "source": "tagged"}


def test_parse_access_lr_scoped_denial_alone_does_not_soften() -> None:
    # An unconditional deny plus a *scoped denial* (redundant, not a carve-out)
    # stays a hard denial — only an allow/ambiguous carve-out softens it.
    restrictions = [
        {"access_type": "denied", "when": {"mode": ["motor_vehicle"]}},
        {
            "access_type": "denied",
            "when": {"mode": ["motor_vehicle"], "during": "Mo-Fr 07:00-19:00"},
        },
    ]
    modes = _mode_map(parse_access_lr(restrictions, road_class="residential").to_dict_list())
    assert modes["motor_vehicle"] == {"value": "denied", "source": "tagged"}


def test_parse_access_lr_unconditional_allow_outranks_deny() -> None:
    # Defensive: if allow and deny co-occur unconditionally, prefer the
    # less separation-signaling reading (allowed), never surface the denial.
    restrictions = [
        {"access_type": "denied", "when": {"mode": ["motor_vehicle"]}},
        {"access_type": "allowed", "when": {"mode": ["motor_vehicle"]}},
    ]
    modes = _mode_map(parse_access_lr(restrictions, road_class="residential").to_dict_list())
    assert modes["motor_vehicle"] == {"value": "allowed", "source": "tagged"}


def test_parse_access_lr_between_subrange_clips() -> None:
    # A tagged restriction scoped to [0, 0.5] applies there; the remainder falls
    # back to the class default (two distinct spans).
    restrictions = [
        {"access_type": "denied", "when": {"mode": ["motor_vehicle"]}, "between": [0.0, 0.5]}
    ]
    lr = parse_access_lr(restrictions, road_class="residential").to_dict_list()
    assert len(lr) == 2
    first, second = lr
    assert first["between"] == [0.0, 0.5]
    assert first["value"]["motor_vehicle"] == {"value": "denied", "source": "tagged"}
    assert second["between"] == [0.5, 1.0]
    assert second["value"]["motor_vehicle"] == {"value": "allowed", "source": "class_default"}


# ---------------------------------------------------------------------------
# (b) class-default pass fills only the allowed inferences, never a denial off
#     a pedestrian/cycleway/footway class
# ---------------------------------------------------------------------------


def test_class_default_driveable_infers_motor_vehicle_allowed() -> None:
    for cls in ("motorway", "primary", "residential", "service"):
        modes = _mode_map(parse_access_lr(None, road_class=cls).to_dict_list())
        assert modes["motor_vehicle"] == {"value": "allowed", "source": "class_default"}


def test_class_default_motorway_infers_bike_foot_denied() -> None:
    # The one safe inferred denial: motorway legally entails bike/foot denied.
    modes = _mode_map(parse_access_lr(None, road_class="motorway").to_dict_list())
    assert modes["motor_vehicle"] == {"value": "allowed", "source": "class_default"}
    assert modes["bicycle"] == {"value": "denied", "source": "class_default"}
    assert modes["foot"] == {"value": "denied", "source": "class_default"}


def test_class_default_cycleway_infers_bike_designated_never_mv_denied() -> None:
    modes = _mode_map(parse_access_lr(None, road_class="cycleway").to_dict_list())
    assert modes["bicycle"] == {"value": "designated", "source": "class_default"}
    # No inferred motor_vehicle denial off a cycleway class (brittleness guard).
    assert "motor_vehicle" not in modes
    assert "foot" not in modes


def test_class_default_footway_infers_foot_designated_never_denial() -> None:
    for cls in ("footway", "sidewalk"):
        modes = _mode_map(parse_access_lr(None, road_class=cls).to_dict_list())
        assert modes["foot"] == {"value": "designated", "source": "class_default"}
        # Never an inferred denial off a pedestrian class.
        assert "motor_vehicle" not in modes
        assert "bicycle" not in modes


def test_class_default_never_emits_inferred_denial_off_pedestrian_classes() -> None:
    # Exhaustive guard: no non-motorway class ever produces a denied value from
    # the class-default pass alone.
    for cls in ("cycleway", "footway", "sidewalk", "pedestrian", "path", "track"):
        modes = _mode_map(parse_access_lr(None, road_class=cls).to_dict_list()) or {}
        assert all(entry["value"] != "denied" for entry in modes.values()), (cls, modes)


def test_unknown_class_yields_no_access() -> None:
    lr = parse_access_lr(None, road_class="pedestrian").to_dict_list()
    # pedestrian has no class default → nothing known → trivial None span.
    assert lr == [{"between": [0.0, 1.0], "value": None}]


# ---------------------------------------------------------------------------
# (c) prompt renders access= with ° for class_default and omits unknown
# ---------------------------------------------------------------------------


def _access_block(road_class: str, restrictions=None) -> dict:
    return {"access_lr": parse_access_lr(restrictions, road_class=road_class).to_dict_list()}


def _prompt_metadata(*, ref_physical: dict, target_physical: dict, target_kind=None) -> dict:
    meta = {
        "group_id": "g1",
        "match_type": "1:1",
        "n_ref_segments": 1,
        "n_target_segments": 1,
        "optimizer_letter": "A",
        "structure": {},
        "same_side_coincidence": {},
        "segments": {
            "reference": [
                {
                    "label": "R3",
                    "id": "r3",
                    "name": "Route de Bellegarde",
                    "class": "secondary",
                    "physical": ref_physical,
                }
            ],
            "target": [
                {
                    "label": "T1",
                    "id": "t1",
                    "name": "Chancy-Bellegarde",
                    "class": "cycleway",
                    "physical": target_physical,
                }
            ],
        },
        "options": [
            {
                "letter": "A",
                "is_optimizer": True,
                "edge_count": 1,
                "total_confidence": 0.9,
                "mean_confidence": 0.9,
                "edges": [{"edge": "R3->T1", "confidence": 0.9}],
            }
        ],
    }
    if target_kind is not None:
        meta["target_kind"] = target_kind
    return meta


def test_summarize_access_marks_class_default_and_unknown() -> None:
    # A secondary road: mv allowed (class_default), bike/foot unknown.
    assert summarize_access(_access_block("secondary")) == "mv:yes° bike:? foot:?"
    # A cycleway target: bike designated (class_default), others unknown.
    assert summarize_access(_access_block("cycleway")) == "mv:? bike:designated° foot:?"


def test_summarize_access_tagged_has_no_degree_mark() -> None:
    block = _access_block("footway", [{"access_type": "allowed", "when": {"mode": ["foot"]}}])
    # foot is tagged (allowed→yes, no °); mv unknown; bicycle unknown.
    assert summarize_access(block) == "mv:? bike:? foot:yes"


def test_edge_struct_renders_tagged_access_omits_class_default() -> None:
    # Edge lines carry a tagged access clause only; class-default-only blocks are
    # suppressed (already shown on the segment line).
    edge = {
        "ref_physical": _access_block(
            "footway", [{"access_type": "allowed", "when": {"mode": ["foot"]}}]
        ),
        "target_physical": _access_block("cycleway"),  # class-default only
    }
    struct = _edge_struct_str(edge)
    assert "R access: mv:? bike:? foot:yes" in struct
    assert "T access:" not in struct


def test_prompt_renders_access_line_and_legend(tmp_path) -> None:
    metadata = _prompt_metadata(
        ref_physical=_access_block("secondary"),
        target_physical=_access_block("cycleway"),
    )
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert "access='mv:yes° bike:? foot:?'" in prompt
    assert "access='mv:? bike:designated° foot:?'" in prompt
    # Legend present, distinguishing class-default from unknown.
    assert "°=class-default" in prompt
    assert "?=unknown" in prompt
    assert "not physical" in prompt


def test_prompt_omits_access_clause_when_no_mode_known(tmp_path) -> None:
    # pedestrian class → no class default → no access clause on that segment.
    metadata = _prompt_metadata(
        ref_physical=_access_block("pedestrian"),
        target_physical=_access_block("pedestrian"),
    )
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert "access='" not in prompt
    # And with no access anywhere, the access legend is suppressed.
    assert "°=class-default" not in prompt


# ---------------------------------------------------------------------------
# (d) route_network header renders for a flagged dataset
# ---------------------------------------------------------------------------

_ROUTE_NETWORK_MARKER = "signed route/itinerary designations"


def test_prompt_renders_route_network_header_when_flagged(tmp_path) -> None:
    metadata = _prompt_metadata(
        ref_physical=_access_block("secondary"),
        target_physical=_access_block("cycleway"),
        target_kind="route_network",
    )
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert _ROUTE_NETWORK_MARKER in prompt
    assert "a route that follows a road matches the road" in prompt


def test_prompt_omits_route_network_header_when_absent(tmp_path) -> None:
    metadata = _prompt_metadata(
        ref_physical=_access_block("secondary"),
        target_physical=_access_block("cycleway"),
    )
    prompt = build_prompt(tmp_path, metadata, {"options": [{"letter": "A"}]})
    assert _ROUTE_NETWORK_MARKER not in prompt
