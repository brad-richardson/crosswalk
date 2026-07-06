"""Assemble + sync the public R2 publication tree (Milestone M5).

The factory (``crosswalk factory run``, M4) produces per-dataset outputs under
``data/factory/release=<overture-release>/dataset=<name>/``. Publishing turns a
selected, *license-cleared* subset of those into the public artifact:

    <staging>/
      index.html                       # credibility page (human)
      index.json                       # machine-readable index (all releases)
      bridges/
        release=<overture-release>/
          index.json                   # per-release index
          checksums.txt                # sha256sum-format manifest
          all_bridges.parquet          # unified long table (+ `dataset` column)
          dataset=<name>/
            bridge.parquet             # copied verbatim from the factory output
            manifest.json              # copied verbatim (provenance)

The tree mirrors the factory partitioning so ``release=`` paths are immutable and
directly range-queryable over HTTPS (DuckDB / parquet-over-HTTP). See
``docs/PUBLISHING.md``.

This module has NO dependency on the matching pipeline — it only reads finished
factory outputs on disk, so it is cheap and safe to run anywhere the factory
tree is available.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from .licenses import LicenseRegistry
from .manifest import MANIFEST_FILENAME, Manifest

BRIDGE_FILENAME = "bridge.parquet"
ALL_BRIDGES_FILENAME = "all_bridges.parquet"
INDEX_JSON = "index.json"
INDEX_HTML = "index.html"
CHECKSUMS_TXT = "checksums.txt"
BRIDGES_PREFIX = "bridges"

# Default public base URL used in the generated query examples. Overridden by
# ``--site-url``; this placeholder makes the examples copy-pasteable in shape.
DEFAULT_SITE_URL = "https://bridges.example.com"

# Overture release used in the generated geometry-join example. Deliberately NOT
# the bridge release: Overture's S3 bucket only keeps recent releases, so old
# release paths 404. GERS ids are stable across releases (~99% measured
# survival), so joining a bridge against a newer Overture release works.
# Check https://docs.overturemaps.org for the current release.
OVERTURE_RELEASE_EXAMPLE = "2026-06-17.0"


# --------------------------------------------------------------------------
# Checksums
# --------------------------------------------------------------------------
def sha256_file(path: Path, _chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 of a file (hex digest)."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for block in iter(lambda: fh.read(_chunk), b""):
            h.update(block)
    return h.hexdigest()


@dataclass(frozen=True)
class FileChecksum:
    sha256: str
    bytes: int

    def to_dict(self) -> dict[str, Any]:
        return {"sha256": self.sha256, "bytes": self.bytes}


def checksum_of(path: Path) -> FileChecksum:
    return FileChecksum(sha256=sha256_file(path), bytes=path.stat().st_size)


# --------------------------------------------------------------------------
# Report structures
# --------------------------------------------------------------------------
@dataclass
class DatasetPublication:
    dataset: str
    release: str
    status: str  # "published" | "excluded"
    display: dict[str, Any] = field(default_factory=dict)
    license: dict[str, Any] | None = None
    reason: str | None = None
    stats: dict[str, Any] = field(default_factory=dict)
    files: dict[str, FileChecksum] = field(default_factory=dict)
    gate: dict[str, Any] | None = None

    @property
    def published(self) -> bool:
        return self.status == "published"

    def to_index(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "status": self.status,
            "display": self.display,
            "stats": self.stats,
        }
        if self.license is not None:
            out["license"] = self.license
        if self.reason is not None:
            out["reason"] = self.reason
        if self.gate is not None:
            out["gate"] = self.gate
        if self.files:
            out["files"] = {name: c.to_dict() for name, c in sorted(self.files.items())}
        return out


@dataclass
class ReleasePublication:
    release: str
    datasets: list[DatasetPublication]
    all_bridges: FileChecksum | None = None
    all_bridges_rows: int = 0

    @property
    def published(self) -> list[DatasetPublication]:
        return [d for d in self.datasets if d.published]

    def to_index(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "datasets": {
                d.dataset: d.to_index() for d in sorted(self.datasets, key=lambda x: x.dataset)
            },
            "n_published": len(self.published),
            "n_excluded": sum(1 for d in self.datasets if not d.published),
        }
        if self.all_bridges is not None:
            out["all_bridges"] = {
                **self.all_bridges.to_dict(),
                "n_rows": self.all_bridges_rows,
            }
        return out


@dataclass
class PublishReport:
    staging_dir: Path
    releases: list[ReleasePublication]
    overture: dict[str, Any]
    site_url: str
    latest_release: str | None
    generated_at: str

    @property
    def n_published(self) -> int:
        return sum(len(r.published) for r in self.releases)

    @property
    def n_excluded(self) -> int:
        return sum(sum(1 for d in r.datasets if not d.published) for r in self.releases)


# --------------------------------------------------------------------------
# Gate floors + display metadata (best-effort context for the credibility page)
# --------------------------------------------------------------------------
def load_gate_floors(mbench_toml: Path) -> dict[str, dict[str, Any]]:
    """Parse ``[gate.<dataset>]`` floor blocks from ``mbench/datasets.toml``.

    Returns ``{dataset: {min_mapped_groups, f1_filtered_floor, exact_filtered_floor}}``.
    These are the *configured* stitch-gate floors (the honest, human-curated
    quality bar), not a live measurement — the credibility page cites them as the
    validation bar where stitching labels exist.
    """
    if not mbench_toml.exists():
        return {}
    data = tomllib.load(mbench_toml.open("rb"))
    return dict(data.get("gate", {}) or {})


def dataset_display(name: str, datasets_dir: Path | None = None) -> dict[str, Any]:
    """Best-effort human display metadata for a dataset from its YAML config."""
    if datasets_dir is None:
        from ..datasets.schema import get_datasets_dir

        datasets_dir = get_datasets_dir()
    yaml_path = datasets_dir / f"{name}.yaml"
    if not yaml_path.exists():
        return {"display_name": name}
    import yaml

    try:
        d = yaml.safe_load(yaml_path.read_text()) or {}
    except Exception:
        return {"display_name": name}
    src = d.get("source") or {}
    return {
        "display_name": d.get("display_name", name),
        "type": d.get("type"),
        "description": d.get("description"),
        "source_type": src.get("type"),
        "source_url": src.get("url"),
    }


# --------------------------------------------------------------------------
# Unified long table
# --------------------------------------------------------------------------
def build_all_bridges(bridge_paths: dict[str, Path], out_path: Path) -> int:
    """Write the per-release unified long table from published bridge parquets.

    Adds a ``dataset`` column and sorts by ``(gers_id, dataset, local_id)`` so
    the file is (a) deterministic and (b) row-group-prunable on ``gers_id`` — the
    reverse-lookup key (given a GERS id, which local datasets reference it).
    Returns the row count. Writes nothing / returns 0 when there is no input.
    """
    import pandas as pd

    if not bridge_paths:
        return 0
    frames = []
    for name, path in sorted(bridge_paths.items()):
        df = pd.read_parquet(path)
        df.insert(0, "dataset", name)
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    sort_cols = [c for c in ("gers_id", "dataset", "local_id") if c in combined.columns]
    combined = combined.sort_values(sort_cols, kind="stable").reset_index(drop=True)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # Deterministic write: no index, stable column order.
    combined.to_parquet(out_path, index=False)
    return len(combined)


# --------------------------------------------------------------------------
# Index (machine-readable)
# --------------------------------------------------------------------------
def build_index(report: PublishReport) -> dict[str, Any]:
    """Top-level machine-readable index across all published releases."""
    return {
        "schema_version": 1,
        "generated_at": report.generated_at,
        "site_url": report.site_url,
        "latest_release": report.latest_release,
        "overture": report.overture,
        "releases": {
            r.release: r.to_index() for r in sorted(report.releases, key=lambda x: x.release)
        },
        "totals": {
            "n_releases": len(report.releases),
            "n_published": report.n_published,
            "n_excluded": report.n_excluded,
        },
    }


def _write_checksums_txt(
    datasets: list[DatasetPublication],
    all_bridges_rel: str | None,
    all_bridges_ck: FileChecksum | None,
    out_path: Path,
) -> None:
    """Write a ``sha256sum``-compatible manifest for a release directory."""
    lines: list[str] = []
    for d in sorted(datasets, key=lambda x: x.dataset):
        for fname, ck in sorted(d.files.items()):
            lines.append(f"{ck.sha256}  dataset={d.dataset}/{fname}")
    if all_bridges_rel and all_bridges_ck:
        lines.append(f"{all_bridges_ck.sha256}  {all_bridges_rel}")
    out_path.write_text("\n".join(lines) + ("\n" if lines else ""))


# --------------------------------------------------------------------------
# Staging assembly
# --------------------------------------------------------------------------
def assemble_staging(
    factory_root: Path,
    staging_dir: Path,
    registry: LicenseRegistry,
    *,
    releases: list[str] | None = None,
    datasets: list[str] | None = None,
    gate_floors: dict[str, dict[str, Any]] | None = None,
    datasets_dir: Path | None = None,
    site_url: str = DEFAULT_SITE_URL,
    generated_at: str | None = None,
    clean: bool = True,
) -> PublishReport:
    """Build the publication staging tree from a factory root.

    Reads every ``release=*/dataset=*`` under ``factory_root`` (optionally filtered
    by ``releases`` / ``datasets``), applies the license registry, copies cleared
    bridge + manifest files into the staging tree, builds the per-release unified
    table + checksums + indexes, and renders the credibility page.

    Deterministic given identical factory inputs (bridge/manifest are copied
    verbatim; the unified table is sorted; only ``generated_at`` is volatile and is
    excluded from the per-file checksums).
    """
    gate_floors = gate_floors or {}
    generated_at = generated_at or datetime.now(UTC).isoformat()
    factory_root = Path(factory_root)
    staging_dir = Path(staging_dir)

    if clean and staging_dir.exists():
        shutil.rmtree(staging_dir)
    staging_dir.mkdir(parents=True, exist_ok=True)

    release_dirs = sorted(factory_root.glob("release=*"))
    if releases is not None:
        wanted = set(releases)
        release_dirs = [d for d in release_dirs if d.name.split("=", 1)[1] in wanted]

    release_pubs: list[ReleasePublication] = []
    for rel_dir in release_dirs:
        release = rel_dir.name.split("=", 1)[1]
        rel_staging = staging_dir / BRIDGES_PREFIX / f"release={release}"
        ds_pubs: list[DatasetPublication] = []
        published_bridges: dict[str, Path] = {}

        for ds_dir in sorted(rel_dir.glob("dataset=*")):
            name = ds_dir.name.split("=", 1)[1]
            if datasets is not None and name not in set(datasets):
                continue
            manifest_path = ds_dir / MANIFEST_FILENAME
            bridge_path = ds_dir / BRIDGE_FILENAME
            if not manifest_path.exists() or not bridge_path.exists():
                # An interrupted / partial factory output — never publish it.
                ds_pubs.append(
                    DatasetPublication(
                        dataset=name,
                        release=release,
                        status="excluded",
                        display=dataset_display(name, datasets_dir),
                        reason="incomplete factory output (missing bridge/manifest)",
                    )
                )
                continue

            m = Manifest.read(manifest_path)
            decision = registry.decision(name)
            stats = _manifest_stats(m)
            gate = gate_floors.get(name)
            display = dataset_display(name, datasets_dir)

            if not decision.approved:
                ds_pubs.append(
                    DatasetPublication(
                        dataset=name,
                        release=release,
                        status="excluded",
                        display=display,
                        license=decision.to_dict(),
                        reason=decision.reason,
                        stats=stats,
                        gate=gate,
                    )
                )
                continue

            # Publish: copy bridge + manifest verbatim into the staging tree.
            out_ds = rel_staging / f"dataset={name}"
            out_ds.mkdir(parents=True, exist_ok=True)
            out_bridge = out_ds / BRIDGE_FILENAME
            out_manifest = out_ds / MANIFEST_FILENAME
            shutil.copyfile(bridge_path, out_bridge)
            shutil.copyfile(manifest_path, out_manifest)
            files = {
                BRIDGE_FILENAME: checksum_of(out_bridge),
                MANIFEST_FILENAME: checksum_of(out_manifest),
            }
            published_bridges[name] = out_bridge
            ds_pubs.append(
                DatasetPublication(
                    dataset=name,
                    release=release,
                    status="published",
                    display=display,
                    license=decision.to_dict(),
                    stats=stats,
                    files=files,
                    gate=gate,
                )
            )

        # Unified long table + per-release index + checksums (only if anything published).
        all_bridges_ck: FileChecksum | None = None
        all_bridges_rows = 0
        if published_bridges:
            all_bridges_path = rel_staging / ALL_BRIDGES_FILENAME
            all_bridges_rows = build_all_bridges(published_bridges, all_bridges_path)
            all_bridges_ck = checksum_of(all_bridges_path)

        rel_pub = ReleasePublication(
            release=release,
            datasets=ds_pubs,
            all_bridges=all_bridges_ck,
            all_bridges_rows=all_bridges_rows,
        )
        release_pubs.append(rel_pub)

        # Per-release outputs are written even if empty-of-published so consumers
        # see the exclusion record; only write files when the dir exists.
        if published_bridges:
            rel_staging.mkdir(parents=True, exist_ok=True)
            (rel_staging / INDEX_JSON).write_text(
                json.dumps(rel_pub.to_index(), indent=2, default=str)
            )
            _write_checksums_txt(
                ds_pubs,
                ALL_BRIDGES_FILENAME if all_bridges_ck else None,
                all_bridges_ck,
                rel_staging / CHECKSUMS_TXT,
            )

    published_releases = [r.release for r in release_pubs if r.published]
    latest = max(published_releases) if published_releases else None

    report = PublishReport(
        staging_dir=staging_dir,
        releases=release_pubs,
        overture=registry.overture,
        site_url=site_url.rstrip("/"),
        latest_release=latest,
        generated_at=generated_at,
    )

    # Top-level index + credibility page.
    (staging_dir / INDEX_JSON).write_text(json.dumps(build_index(report), indent=2, default=str))
    (staging_dir / INDEX_HTML).write_text(render_credibility_page(report))
    return report


def _manifest_stats(m: Manifest) -> dict[str, Any]:
    n_target = m.n_target or 0
    match_rate = round(m.n_matched / n_target, 4) if n_target else None
    return {
        "release": m.release,
        "created_at": m.created_at,
        "feature_version": m.feature_version,
        "buffer_distance_m": m.buffer_distance_m,
        "method": m.method,
        "n_reference": m.n_reference,
        "n_target": m.n_target,
        "n_candidates": m.n_candidates,
        "n_matched": m.n_matched,
        "n_review": m.n_review,
        "n_unmatched": m.n_unmatched,
        "match_rate": match_rate,
        "n_groups": (m.groups or {}).get("n_groups", 0),
        "n_m_to_n": (m.groups or {}).get("n_m_to_n", 0),
        "n_oversized": (m.groups or {}).get("n_oversized", 0),
        "wall_s": m.wall_s,
    }


# --------------------------------------------------------------------------
# Credibility page (static, self-contained HTML)
# --------------------------------------------------------------------------
def render_credibility_page(report: PublishReport) -> str:
    """Render the self-contained static credibility page for the staging root."""
    from html import escape

    ov = report.overture
    site = report.site_url
    latest = report.latest_release

    def pct(x: Any) -> str:
        return f"{100 * x:.1f}%" if isinstance(x, (int, float)) else "—"

    rows_html: list[str] = []
    excluded_html: list[str] = []
    for rel in sorted(report.releases, key=lambda r: r.release, reverse=True):
        for d in sorted(rel.datasets, key=lambda x: x.dataset):
            disp = d.display or {}
            name = escape(d.dataset)
            dname = escape(str(disp.get("display_name") or d.dataset))
            dtype = escape(str(disp.get("type") or "—"))
            if d.published:
                s = d.stats
                lic = escape(str((d.license or {}).get("license") or "—"))
                gate = d.gate
                gate_txt = (
                    f"F1≥{gate.get('f1_filtered_floor')}, exact≥{gate.get('exact_filtered_floor')}"
                    if gate
                    else "not yet gated"
                )
                rows_html.append(
                    f"<tr><td><code>{name}</code><div class='sub'>{dname}</div></td>"
                    f"<td>{dtype}</td><td>{escape(rel.release)}</td>"
                    f"<td class='num'>{s.get('n_target', 0):,}</td>"
                    f"<td class='num'>{s.get('n_matched', 0):,}</td>"
                    f"<td class='num'>{pct(s.get('match_rate'))}</td>"
                    f"<td class='num'>{s.get('n_review', 0):,}</td>"
                    f"<td class='num'>{s.get('n_groups', 0):,}</td>"
                    f"<td class='num'>{s.get('wall_s') or '—'}</td>"
                    f"<td>{lic}</td><td class='gate'>{escape(gate_txt)}</td></tr>"
                )
            else:
                reason = escape(str(d.reason or "excluded"))
                excluded_html.append(
                    f"<tr><td><code>{name}</code><div class='sub'>{dname}</div></td>"
                    f"<td>{dtype}</td><td>{reason}</td></tr>"
                )

    # Raw placeholders — the whole query string is HTML-escaped exactly once below.
    example_release = latest or "<release>"
    published_ds = sorted({d.dataset for r in report.releases for d in r.published})
    example_ds = (
        "us_montana_missoula"
        if "us_montana_missoula" in published_ds
        else (published_ds[0] if published_ds else "<dataset>")
    )
    q_dataset = (
        f"SELECT * FROM read_parquet(\n"
        f"  '{site}/bridges/release={example_release}/dataset={example_ds}/bridge.parquet'\n"
        f")\nWHERE match_decision = 'match';"
    )
    q_reverse = (
        f"SELECT dataset, local_id, confidence, match_type, match_decision\n"
        f"FROM read_parquet(\n"
        f"  '{site}/bridges/release={example_release}/all_bridges.parquet'\n"
        f")\nWHERE gers_id = '<gers-id>';"
    )
    q_join = (
        f"INSTALL spatial; LOAD spatial; INSTALL httpfs; LOAD httpfs;\n"
        f"SET s3_region = 'us-west-2';\n"
        f"WITH bridge AS (\n"
        f"  SELECT * FROM read_parquet(\n"
        f"    '{site}/bridges/release={example_release}/dataset={example_ds}/bridge.parquet'\n"
        f"  )\n"
        f"  WHERE match_decision = 'match'\n"
        f")\n"
        f"SELECT b.local_id, b.gers_id, b.confidence, ST_AsText(s.geometry) AS overture_wkt\n"
        f"FROM bridge b\n"
        f"JOIN read_parquet(\n"
        f"  -- Overture release: check https://docs.overturemaps.org for the latest.\n"
        f"  -- GERS ids are stable across releases, so it need not match the bridge release.\n"
        f"  's3://overturemaps-us-west-2/release/{OVERTURE_RELEASE_EXAMPLE}"
        f"/theme=transportation/type=segment/*',\n"
        f"  hive_partitioning=true\n"
        f") s ON s.id = b.gers_id\n"
        f"-- bbox: Missoula, MT (us_montana_missoula). Swap bbox + dataset together —\n"
        f"-- e.g. Seattle: xmin -122.44 / -122.22, ymin 47.49 / 47.74.\n"
        f"WHERE s.bbox.xmin BETWEEN -114.41 AND -113.68\n"
        f"  AND s.bbox.ymin BETWEEN 46.72 AND 47.14\n"
        f"LIMIT 100;"
    )

    published_table = (
        "<table><thead><tr>"
        "<th>dataset</th><th>type</th><th>release</th><th>targets</th><th>matched</th>"
        "<th>match rate</th><th>review</th><th>groups</th><th>wall s</th>"
        "<th>license</th><th>quality gate</th>"
        "</tr></thead><tbody>" + "".join(rows_html) + "</tbody></table>"
        if rows_html
        else "<p class='empty'>No datasets published in this staging build.</p>"
    )
    excluded_table = (
        "<table><thead><tr><th>dataset</th><th>type</th><th>reason</th></tr></thead>"
        "<tbody>" + "".join(excluded_html) + "</tbody></table>"
        if excluded_html
        else "<p class='empty'>None.</p>"
    )

    ov_attr = escape(str(ov.get("attribution", "")))
    ov_url = escape(str(ov.get("url", "https://overturemaps.org/")))
    generated = escape(report.generated_at)
    latest_txt = escape(str(latest or "—"))

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>GERS Bridge Tables — crosswalk</title>
<style>
  :root {{ color-scheme: light dark; --fg:#1a1a1a; --muted:#666; --bg:#fff;
           --line:#e2e2e2; --accent:#2b6cb0; --code:#f4f4f6; }}
  @media (prefers-color-scheme: dark) {{
    :root {{ --fg:#e6e6e6; --muted:#9a9a9a; --bg:#141414; --line:#2c2c2c;
             --accent:#6ea8fe; --code:#1e1e22; }}
  }}
  * {{ box-sizing: border-box; }}
  body {{ font: 15px/1.6 -apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,sans-serif;
          color:var(--fg); background:var(--bg); margin:0; padding:2rem 1.25rem; }}
  main {{ max-width: 1080px; margin: 0 auto; }}
  h1 {{ font-size: 1.7rem; margin:0 0 .25rem; }}
  h2 {{ font-size: 1.2rem; margin:2.2rem 0 .6rem; border-bottom:1px solid var(--line);
        padding-bottom:.3rem; }}
  .lead {{ color:var(--muted); margin:.2rem 0 1rem; }}
  code {{ background:var(--code); padding:.1rem .3rem; border-radius:4px;
          font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.85em; }}
  pre {{ background:var(--code); padding:1rem; border-radius:8px; overflow-x:auto;
         font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:.82rem; }}
  .tablewrap {{ overflow-x:auto; }}
  table {{ border-collapse:collapse; width:100%; font-size:.85rem; margin:.5rem 0; }}
  th,td {{ text-align:left; padding:.45rem .6rem; border-bottom:1px solid var(--line);
           vertical-align:top; }}
  th {{ color:var(--muted); font-weight:600; white-space:nowrap; }}
  td.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
  td.gate {{ color:var(--muted); }}
  .sub {{ color:var(--muted); font-size:.9em; }}
  .empty {{ color:var(--muted); }}
  .pill {{ display:inline-block; background:var(--code); border-radius:999px;
           padding:.1rem .6rem; font-size:.8rem; color:var(--muted); }}
  .caveat {{ border-left:3px solid var(--accent); padding:.2rem 0 .2rem .9rem;
             margin:.6rem 0; color:var(--fg); }}
  details.methodology {{ margin:2.2rem 0 .6rem; }}
  details.methodology summary {{ cursor:pointer; color:var(--accent);
             font-weight:600; }}
  footer {{ margin-top:2.5rem; color:var(--muted); font-size:.8rem;
            border-top:1px solid var(--line); padding-top:1rem; }}
  a {{ color:var(--accent); }}
</style>
</head>
<body><main>
  <h1>GERS Bridge Tables</h1>
  <p class="lead">Look up how a city's own street and path IDs map to
  <strong>Overture Maps GERS ids</strong>. Each city gets a bridge table — a plain
  Parquet file connecting the two ID schemes — so anything keyed to local IDs can be
  joined to the open map with one line of SQL. Built by
  <a href="https://github.com/brad-richardson/matcher">crosswalk</a> and regenerated
  for each Overture release.</p>
  <p class="lead">This is an early work in progress — coverage grows city by city.
  Spot a problem or want your city added?
  <a href="https://github.com/brad-richardson/matcher/issues">Feedback is welcome</a>.</p>
  <p class="lead">An independent community project — not affiliated with the Overture
  Maps Foundation.</p>
  <p><span class="pill">latest release: {latest_txt}</span>
     <span class="pill">generated: {generated}</span></p>

  <h2>Query it</h2>
  <p>Every table is a plain Parquet file — point DuckDB (or any Parquet-over-HTTP
  reader) straight at the URL. No API, no signup, and you only download the parts of
  the file your query touches.</p>
  <p><strong>All matches for one dataset:</strong></p>
  <pre>{escape(q_dataset)}</pre>
  <p><strong>Reverse lookup — which local datasets reference a GERS id</strong>
  (searches every published dataset in a release at once):</p>
  <pre>{escape(q_reverse)}</pre>
  <p><strong>Join to Overture geometry</strong> (DuckDB CLI). Bridge tables carry IDs
  only — join <code>gers_id</code> to Overture's transportation theme, read straight
  from Overture's public S3, to get geometry. The theme covers the whole planet, but
  the bbox filter lets DuckDB skip almost all of it — only the parts covering your
  city are downloaded:</p>
  <pre>{escape(q_join)}</pre>

  <h2>Published datasets</h2>
  <p class="lead"><strong>matched</strong> counts confident matches only — borderline
  candidates ship in each table marked <code>review</code>, but don't count as
  matches. Counts come from each dataset's <code>manifest.json</code>, published
  alongside the bridge as its provenance record.</p>
  <div class="tablewrap">{published_table}</div>

  <h2>Waiting on license review</h2>
  <p class="lead">We only publish data whose license we've verified. These datasets
  are ready but on hold until their source license checks out.</p>
  <div class="tablewrap">{excluded_table}</div>

  <details class="methodology">
  <summary>Methodology notes</summary>
  <div class="caveat"><strong>What counts as a match.</strong> Only
  <code>match_decision = 'match'</code> rows are the bridge; <code>review</code> rows
  ship in the table but are lower-confidence and excluded from headline match rates.</div>
  <div class="caveat"><strong>Confidence scores.</strong> The <code>confidence</code>
  column is a calibrated probability of a correct match — filter to whatever threshold
  suits your use case.</div>
  <div class="caveat"><strong>Quality gate coverage is partial.</strong> The quality
  gate checks how well many-to-many match groups are assembled, but only where curated
  review labels exist; datasets showing "not yet gated" haven't had that check.</div>
  <div class="caveat"><strong>Per-dataset validation varies.</strong> Match quality
  is benchmarked on a few datasets (see the repo's <code>BENCHMARK_RESULTS.md</code>);
  treat un-benchmarked datasets as provisional.</div>
  </details>

  <h2>Licensing &amp; attribution</h2>
  <p>Each published table is a derived work of both the local source dataset and
  Overture. Redistribution must carry both attributions.</p>
  <p><strong>Overture:</strong> {ov_attr} (<a href="{ov_url}">{ov_url}</a>)</p>
  <p><strong>Per-dataset source license</strong> is shown in the table above and in
  each dataset's entry in <code>index.json</code>. We only publish data whose license
  we've verified (see the on-hold list above).</p>

  <footer>
    Machine-readable index: <a href="{escape(site)}/index.json"><code>index.json</code></a> ·
    per-release <code>checksums.txt</code> (sha256) accompanies every release.
    Generated by <code>crosswalk factory publish</code>.
  </footer>
</main></body>
</html>
"""
