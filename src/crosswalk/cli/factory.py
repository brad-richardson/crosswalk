"""``crosswalk factory`` — the bridge-table factory command group (Milestone M4).

Batch, versioned, resumable stitching of many datasets to Overture, with a
scored-candidate cache for fast re-optimization and a per-release GERS churn
delta report. See ``docs/FACTORY.md``.
"""

from __future__ import annotations

import os
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

console = Console()

factory_app = typer.Typer(
    name="factory",
    help="Bridge-table factory: batch, versioned, resumable dataset stitching (M4).",
    no_args_is_help=True,
)


def _default_workers() -> int:
    """Conservative local default: min(4, cores//4), at least 1.

    On the always-on 20-core box, override to 12 (the media-server core budget) —
    see docs/FACTORY.md. Each worker scores single-threaded (``--jobs-per-dataset
    1``) so ``workers`` is the total core budget.
    """
    cores = os.cpu_count() or 4
    return max(1, min(4, cores // 4))


def _warn_if_nested_pools(workers: int) -> None:
    """Warn when ``--workers > 1``: the outer dataset ``ProcessPoolExecutor`` nests
    inside each dataset's fork-based feature-scoring pool, which crashes with
    ``BrokenProcessPool`` on large sweeps (see docs/FACTORY.md box runbook). Least
    invasive honest guard — we don't restructure the multiprocessing here (deferred
    follow-up); we just steer callers to ``--workers 1 --jobs-per-dataset N``.
    """
    if workers > 1:
        console.print(
            f"[yellow]WARNING: --workers={workers} (>1) nests fork-based process "
            "pools (outer dataset pool x inner scoring pool) and can crash with "
            "BrokenProcessPool on large sweeps. Prefer --workers 1 "
            "--jobs-per-dataset N (see docs/FACTORY.md box runbook).[/yellow]"
        )


def _resolve_paths(raw_dir: Path | None, output_dir: Path | None):
    from ..factory import FactoryPaths
    from ..filenames import PROJECT_ROOT

    raw = raw_dir or (PROJECT_ROOT / "data" / "raw")
    root = output_dir or (PROJECT_ROOT / "data" / "factory")
    return raw, FactoryPaths(root=root)


def _select_pairs(raw_dir: Path, all_datasets: bool, datasets: list[str] | None):
    from ..factory import discover_pairs

    names = None if all_datasets else (datasets or None)
    if not all_datasets and not names:
        console.print("[red]Provide dataset name(s) as arguments / -D, or --all.[/red]")
        raise typer.Exit(1)
    pairs = discover_pairs(raw_dir=raw_dir, names=names)
    if not pairs:
        console.print(f"[yellow]No stitchable datasets found under {raw_dir}[/yellow]")
        raise typer.Exit(0)
    return pairs


def _render_summary(summaries: list[dict], title: str) -> int:
    """Render the run summary table; return the number of failures."""
    table = Table(title=title, show_lines=False)
    table.add_column("dataset", style="cyan", no_wrap=True)
    table.add_column("release")
    table.add_column("status")
    table.add_column("wall_s", justify="right")
    table.add_column("matched", justify="right")
    table.add_column("review", justify="right")
    table.add_column("unmatched", justify="right")
    table.add_column("groups", justify="right")
    table.add_column("oversized", justify="right")

    n_fail = 0
    for s in sorted(summaries, key=lambda x: x["dataset"]):
        status = s["status"]
        if status == "failed":
            n_fail += 1
            style = "red"
        elif status == "skipped":
            style = "yellow"
        else:
            style = "green"
        matched = s.get("n_matched", "")
        n_target = s.get("n_target")
        matched_str = (
            f"{matched} ({100 * matched / n_target:.1f}%)"
            if isinstance(matched, int) and n_target
            else str(matched)
        )
        table.add_row(
            s["dataset"],
            str(s.get("release") or "-"),
            f"[{style}]{status}[/{style}]",
            str(s.get("wall_s", "")),
            matched_str,
            str(s.get("n_review", "")),
            str(s.get("n_unmatched", "")),
            str(s.get("n_groups", "")),
            str(s.get("n_oversized", "")),
        )
    console.print(table)
    for s in summaries:
        if s["status"] == "failed":
            console.print(f"[red]  {s['dataset']}: {s.get('error', 'unknown error')}[/red]")
    return n_fail


@factory_app.command()
def run(
    datasets: list[str] = typer.Argument(None, help="Dataset names to run (or use --all)."),
    all_datasets: bool = typer.Option(False, "--all", "-a", help="Run all stitchable datasets."),
    dataset_opt: list[str] = typer.Option(
        None, "--dataset", "-D", help="Dataset name (repeatable); merged with positional args."
    ),
    workers: int = typer.Option(
        None,
        "--workers",
        "-w",
        help=(
            "Concurrent dataset worker processes (default: min(4, cores/4)). "
            "WARNING: keep at 1 on the box — >1 nests fork-based process pools "
            "(outer dataset pool x inner scoring pool) and crashes with "
            "BrokenProcessPool on large sweeps; put cores into -j instead."
        ),
    ),
    jobs_per_dataset: int = typer.Option(
        1,
        "--jobs-per-dataset",
        "-j",
        help="Internal scoring parallelism per dataset process (use 12 on the box).",
    ),
    release: str = typer.Option(
        None,
        "--release",
        help="Override the Overture release (else derived from segments .meta.yaml).",
    ),
    buffer_m: float = typer.Option(75.0, "--buffer-m", "-b", help="Candidate search radius (m)."),
    force: bool = typer.Option(
        False, "--force", "-f", help="Rerun even if the manifest is current."
    ),
    raw_dir: Path = typer.Option(None, "--raw-dir", help="Raw-data dir (default: data/raw)."),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Factory root (default: data/factory)."
    ),
):
    """Run the full stitch pipeline for one or more datasets, versioned + resumable."""
    from ..factory.runner import run_batch

    raw, paths = _resolve_paths(raw_dir, output_dir)
    w = workers if workers is not None else _default_workers()
    _warn_if_nested_pools(w)
    merged = list(datasets or []) + list(dataset_opt or [])
    pairs = _select_pairs(raw, all_datasets, merged)

    console.print(
        f"[blue]Factory run: {len(pairs)} dataset(s), workers={w}, "
        f"jobs/dataset={jobs_per_dataset}, root={paths.root}[/blue]"
    )
    summaries = run_batch(
        pairs,
        paths,
        release_override=release,
        workers=w,
        buffer_distance_m=buffer_m,
        n_jobs=jobs_per_dataset,
        force=force,
    )
    n_fail = _render_summary(summaries, "Factory run summary")
    raise typer.Exit(1 if n_fail else 0)


@factory_app.command()
def reoptimize(
    datasets: list[str] = typer.Argument(None, help="Dataset names to reoptimize (or use --all)."),
    all_datasets: bool = typer.Option(
        False, "--all", "-a", help="Reoptimize all stitchable datasets."
    ),
    dataset_opt: list[str] = typer.Option(
        None, "--dataset", "-D", help="Dataset name (repeatable)."
    ),
    workers: int = typer.Option(None, "--workers", "-w", help="Concurrent worker processes."),
    release: str = typer.Option(None, "--release", help="Override the Overture release."),
    buffer_m: float = typer.Option(75.0, "--buffer-m", "-b", help="Candidate search radius (m)."),
    raw_dir: Path = typer.Option(None, "--raw-dir", help="Raw-data dir (default: data/raw)."),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Factory root (default: data/factory)."
    ),
):
    """Re-run grouping/optimization/sidecar from the cached scored candidates (~2 s).

    Fast-path for iterating on grouping/optimizer settings without re-scoring.
    Requires a prior ``factory run`` whose scored cache is still valid.
    """
    from ..factory.runner import run_batch

    raw, paths = _resolve_paths(raw_dir, output_dir)
    merged = list(datasets or []) + list(dataset_opt or [])
    pairs = _select_pairs(raw, all_datasets, merged)
    w = workers if workers is not None else _default_workers()

    console.print(f"[blue]Factory reoptimize: {len(pairs)} dataset(s), root={paths.root}[/blue]")
    summaries = run_batch(
        pairs,
        paths,
        release_override=release,
        workers=w,
        buffer_distance_m=buffer_m,
        reoptimize=True,
    )
    n_fail = _render_summary(summaries, "Factory reoptimize summary")
    raise typer.Exit(1 if n_fail else 0)


@factory_app.command()
def delta(
    dataset: str = typer.Argument(..., help="Dataset name."),
    from_release: str = typer.Option(..., "--from", help="Baseline release identifier."),
    to_release: str = typer.Option(..., "--to", help="Comparison release identifier."),
    fmt: str = typer.Option("md", "--format", help="Output format: md | csv."),
    output: Path = typer.Option(None, "--output", "-o", help="Write to file instead of stdout."),
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Factory root (default: data/factory)."
    ),
):
    """Report GERS match churn for a dataset between two factory releases."""
    from ..factory.delta import compute_delta

    _, paths = _resolve_paths(None, output_dir)
    from_bridge = paths.bridge(from_release, dataset)
    to_bridge = paths.bridge(to_release, dataset)
    for label, p in (("from", from_bridge), ("to", to_bridge)):
        if not p.exists():
            console.print(f"[red]Missing {label} bridge: {p}[/red]")
            raise typer.Exit(1)

    result = compute_delta(dataset, from_bridge, to_bridge, from_release, to_release)
    s = result.summary
    console.print(
        f"[blue]{dataset}: same={s['same']} changed={s['changed']} "
        f"lost={s['lost']} gained={s['gained']}[/blue]"
    )

    if fmt == "csv":
        content = result.details.to_csv(index=False)
    elif fmt == "md":
        content = result.to_markdown()
    else:
        console.print(f"[red]Unknown format '{fmt}' (use md | csv).[/red]")
        raise typer.Exit(1)

    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(content)
        console.print(f"[green]Wrote {fmt} delta to {output}[/green]")
    else:
        console.print(content)


def _render_publish_summary(report, target_desc: str) -> None:
    """Render the publish staging summary (published + excluded per release)."""
    table = Table(title=f"Publish staging ({report.staging_dir})", show_lines=False)
    table.add_column("release")
    table.add_column("dataset", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("matched", justify="right")
    table.add_column("match%", justify="right")
    table.add_column("license / reason")
    for rel in sorted(report.releases, key=lambda r: r.release):
        for d in sorted(rel.datasets, key=lambda x: x.dataset):
            if d.published:
                s = d.stats
                mr = s.get("match_rate")
                lic = (d.license or {}).get("license") or "-"
                table.add_row(
                    rel.release,
                    d.dataset,
                    "[green]published[/green]",
                    str(s.get("n_matched", "")),
                    f"{100 * mr:.1f}%" if isinstance(mr, (int, float)) else "-",
                    str(lic),
                )
            else:
                table.add_row(
                    rel.release,
                    d.dataset,
                    "[yellow]excluded[/yellow]",
                    "",
                    "",
                    str(d.reason or "excluded"),
                )
    console.print(table)
    console.print(
        f"[blue]{report.n_published} published, {report.n_excluded} excluded "
        f"(pending review); latest_release={report.latest_release}; target={target_desc}[/blue]"
    )


def _render_targets_publish_summary(report, target_desc: str) -> None:
    """Render the target-snapshot publish staging summary (published + excluded)."""
    table = Table(title=f"Publish staging targets ({report.staging_dir})", show_lines=False)
    table.add_column("dataset", style="cyan", no_wrap=True)
    table.add_column("status")
    table.add_column("snapshot")
    table.add_column("size", justify="right")
    table.add_column("provenance")
    table.add_column("license / reason")
    for d in sorted(report.datasets, key=lambda x: x.dataset):
        if d.published:
            size_mb = f"{d.size_bytes / 1e6:.2f} MB" if d.size_bytes else "-"
            lic = (d.license or {}).get("license") or "-"
            table.add_row(
                d.dataset,
                "[green]published[/green]",
                d.snapshot or "-",
                size_mb,
                d.provenance_from or "-",
                str(lic),
            )
        else:
            table.add_row(
                d.dataset, "[yellow]excluded[/yellow]", "", "", "", str(d.reason or "excluded")
            )
    console.print(table)
    console.print(
        f"[blue]{report.n_published} published, {report.n_excluded} excluded "
        f"(pending review); target={target_desc}[/blue]"
    )


@factory_app.command()
def publish(
    datasets: list[str] = typer.Argument(
        None, help="Dataset names to publish (default: all cleared)."
    ),
    all_datasets: bool = typer.Option(
        False,
        "--all",
        "-a",
        help="Publish all discovered datasets (explicit alias of the default; "
        "overrides any positional/-D names).",
    ),
    dataset_opt: list[str] = typer.Option(
        None, "--dataset", "-D", help="Dataset name (repeatable)."
    ),
    release: list[str] = typer.Option(
        None, "--release", help="Restrict to these release(s) (repeatable; default: all present)."
    ),
    targets: bool = typer.Option(
        False,
        "--targets",
        help="Publish TARGET dataset snapshots (data/raw/<name>_v1.0.parquet) under the "
        "R2 'targets/' prefix, instead of bridge tables under 'bridges/'.",
    ),
    raw_dir: Path = typer.Option(
        None,
        "--raw-dir",
        help="Raw-data dir to publish target snapshots from (default: "
        "data/raw). Only used with --targets.",
    ),
    target_dir: Path = typer.Option(
        None, "--target-dir", help="Publish to a LOCAL directory (no credentials). Omit for R2."
    ),
    dry_run: bool = typer.Option(
        True,
        "--dry-run/--no-dry-run",
        help="Build the staging tree + report what WOULD upload, without syncing (default: on).",
    ),
    force: bool = typer.Option(
        False,
        "--force",
        "-f",
        help="Overwrite an already-published (immutable) release/snapshot.",
    ),
    site_url: str = typer.Option(
        None, "--site-url", help="Public base URL used in the credibility page query examples."
    ),
    staging_dir: Path = typer.Option(
        None,
        "--staging-dir",
        help="Where to build the staging tree (default: data/publish_staging, or "
        "data/publish_staging_targets with --targets).",
    ),
    factory_root: Path = typer.Option(
        None, "--factory-root", help="Factory output root to publish from (default: data/factory)."
    ),
):
    """Assemble the public R2 publication tree and sync it.

    Two modes:

    * Bridge tables (default): builds a deterministic staging tree (bridge +
      manifest per cleared dataset, a per-release unified ``all_bridges.parquet``,
      checksums, machine-readable ``index.json``, and the credibility
      ``index.html``) under the ``bridges/`` prefix. Immutable per ``release=``.
    * Target snapshots (``--targets``): builds a staging tree of local target
      dataset snapshots (``data/raw/<name>_v1.0.parquet`` + a normalized
      ``meta.yaml`` provenance sidecar + ``latest.json``/``index.json``) under
      the ``targets/`` prefix. Immutable per ``dataset=*/snapshot=*``.

    Both modes exclude license-unverified datasets, and then, unless
    ``--dry-run`` (the default), sync to a local directory (``--target-dir``) or
    to Cloudflare R2 (S3-compatible ``aws`` CLI, credentials from ``R2_*`` env
    vars). An already-published release/snapshot is skipped (never overwritten)
    unless ``--force``.
    """
    from ..factory.licenses import LicenseRegistry
    from ..factory.publish import (
        DEFAULT_SITE_URL,
        assemble_staging,
        load_gate_floors,
    )
    from ..factory.publish_sync import (
        R2_ENV_VARS,
        build_aws_sync_argv,
        missing_r2_env,
        r2_env,
        staged_files,
        sync_local,
        sync_r2,
        sync_targets_local,
        sync_targets_r2,
    )
    from ..factory.publish_targets import assemble_targets_staging
    from ..filenames import PROJECT_ROOT

    try:
        registry = LicenseRegistry.load()
    except FileNotFoundError as exc:
        console.print(f"[red]{exc}[/red]")
        raise typer.Exit(1) from exc

    # Default (no names) = all cleared datasets; --all is the explicit alias and
    # overrides any names passed alongside it.
    merged = list(datasets or []) + list(dataset_opt or [])
    ds_filter = None if all_datasets else (merged or None)

    # --- Target snapshots ---
    if targets:
        raw = raw_dir or (PROJECT_ROOT / "data" / "raw")
        staging = staging_dir or (PROJECT_ROOT / "data" / "publish_staging_targets")

        report = assemble_targets_staging(
            raw_dir=raw,
            staging_dir=staging,
            registry=registry,
            datasets=ds_filter,
        )

        target_desc = f"local:{target_dir}" if target_dir is not None else "r2"
        _render_targets_publish_summary(report, target_desc)

        if report.n_published == 0:
            console.print(
                "[yellow]Nothing cleared to publish — staging tree built for review only.[/yellow]"
            )

        if target_dir is not None:
            if dry_run:
                files = staged_files(staging)
                console.print(
                    f"[blue]DRY RUN: would copy {len(files)} file(s) from {staging} "
                    f"to {target_dir}. Re-run with --no-dry-run to write.[/blue]"
                )
                raise typer.Exit(0)
            written, plan = sync_targets_local(staging, target_dir, force=force)
            if plan.skipped_snapshots:
                skipped = ", ".join(f"{ds}@{snap}" for ds, snap in plan.skipped_snapshots)
                console.print(
                    f"[yellow]Skipped already-published (immutable) snapshot(s): "
                    f"{skipped} — use --force to replace.[/yellow]"
                )
            console.print(f"[green]Published {len(written)} file(s) to {target_dir}[/green]")
            raise typer.Exit(0)

        cfg = r2_env()
        if cfg is None:
            missing = ", ".join(missing_r2_env())
            console.print(
                f"[yellow]R2 credentials absent (missing: {missing}). Forcing dry run.[/yellow]"
            )
            files = staged_files(staging)
            console.print(
                f"[blue]DRY RUN: staging tree built at {staging} ({len(files)} file(s)). "
                f"To publish to R2, set {', '.join(R2_ENV_VARS.values())} and re-run "
                "with --no-dry-run.[/blue]"
            )
            raise typer.Exit(0)
        if dry_run:
            argv = build_aws_sync_argv(staging, cfg)
            console.print(
                f"[blue]DRY RUN: would run: {' '.join(argv)}\n"
                "(already-published snapshots are excluded at upload time — immutable). "
                "Re-run with --no-dry-run to upload.[/blue]"
            )
            raise typer.Exit(0)
        try:
            _, plan = sync_targets_r2(staging, cfg, force=force)
        except Exception as exc:  # aws CLI failure or a failed-closed existence check
            console.print(f"[red]R2 sync failed: {exc}[/red]")
            raise typer.Exit(1) from exc
        if plan.skipped_snapshots:
            skipped = ", ".join(f"{ds}@{snap}" for ds, snap in plan.skipped_snapshots)
            console.print(
                f"[yellow]Skipped already-published (immutable) snapshot(s): "
                f"{skipped} — use --force to replace.[/yellow]"
            )
        console.print(
            f"[green]Published staging tree to R2 bucket '{cfg.bucket}' (targets/).[/green]"
        )
        raise typer.Exit(0)

    # --- Bridge tables ---
    root = factory_root or (PROJECT_ROOT / "data" / "factory")
    staging = staging_dir or (PROJECT_ROOT / "data" / "publish_staging")
    if not root.exists():
        console.print(
            f"[red]No factory output at {root} — run 'crosswalk factory run' first.[/red]"
        )
        raise typer.Exit(1)

    gate_floors = load_gate_floors(PROJECT_ROOT / "mbench" / "datasets.toml")
    report = assemble_staging(
        factory_root=root,
        staging_dir=staging,
        registry=registry,
        releases=list(release) if release else None,
        datasets=ds_filter,
        gate_floors=gate_floors,
        site_url=site_url or DEFAULT_SITE_URL,
    )

    # Decide the sync target for the summary + execution.
    if target_dir is not None:
        target_desc = f"local:{target_dir}"
    else:
        target_desc = "r2"
    _render_publish_summary(report, target_desc)

    if report.n_published == 0:
        console.print(
            "[yellow]Nothing cleared to publish — staging tree built for review only.[/yellow]"
        )

    # --- Sync ---
    if target_dir is not None:
        if dry_run:
            files = staged_files(staging)
            console.print(
                f"[blue]DRY RUN: would copy {len(files)} file(s) from {staging} "
                f"to {target_dir}. Re-run with --no-dry-run to write.[/blue]"
            )
            raise typer.Exit(0)
        written, plan = sync_local(staging, target_dir, force=force)
        if plan.skipped_releases:
            console.print(
                f"[yellow]Skipped already-published (immutable) release(s): "
                f"{', '.join(plan.skipped_releases)} — use --force to replace.[/yellow]"
            )
        console.print(f"[green]Published {len(written)} file(s) to {target_dir}[/green]")
        raise typer.Exit(0)

    # R2 target.
    cfg = r2_env()
    if cfg is None:
        missing = ", ".join(missing_r2_env())
        console.print(
            f"[yellow]R2 credentials absent (missing: {missing}). Forcing dry run.[/yellow]"
        )
        files = staged_files(staging)
        console.print(
            f"[blue]DRY RUN: staging tree built at {staging} ({len(files)} file(s)). "
            f"To publish to R2, set {', '.join(R2_ENV_VARS.values())} and re-run "
            "with --no-dry-run.[/blue]"
        )
        raise typer.Exit(0)
    if dry_run:
        argv = build_aws_sync_argv(staging, cfg)
        console.print(
            f"[blue]DRY RUN: would run: {' '.join(argv)}\n"
            "(already-published releases are excluded at upload time — immutable). "
            "Re-run with --no-dry-run to upload.[/blue]"
        )
        raise typer.Exit(0)
    try:
        _, plan = sync_r2(staging, cfg, force=force)
    except Exception as exc:  # aws CLI failure or a failed-closed existence check
        console.print(f"[red]R2 sync failed: {exc}[/red]")
        raise typer.Exit(1) from exc
    if plan.skipped_releases:
        console.print(
            f"[yellow]Skipped already-published (immutable) release(s): "
            f"{', '.join(plan.skipped_releases)} — use --force to replace.[/yellow]"
        )
    console.print(f"[green]Published staging tree to R2 bucket '{cfg.bucket}'.[/green]")


@factory_app.command()
def status(
    output_dir: Path = typer.Option(
        None, "--output-dir", help="Factory root (default: data/factory)."
    ),
):
    """List factory releases and per-dataset manifest summaries."""
    from ..factory.manifest import Manifest

    _, paths = _resolve_paths(None, output_dir)
    if not paths.root.exists():
        console.print(f"[yellow]No factory output at {paths.root}[/yellow]")
        raise typer.Exit(0)

    table = Table(title=f"Factory status ({paths.root})")
    table.add_column("release")
    table.add_column("dataset", style="cyan")
    table.add_column("matched", justify="right")
    table.add_column("groups", justify="right")
    table.add_column("wall_s", justify="right")
    table.add_column("created_at")

    for rel_dir in sorted(paths.root.glob("release=*")):
        release = rel_dir.name.split("=", 1)[1]
        for ds_dir in sorted(rel_dir.glob("dataset=*")):
            name = ds_dir.name.split("=", 1)[1]
            mpath = ds_dir / "manifest.json"
            if not mpath.exists():
                table.add_row(release, name, "-", "-", "-", "(no manifest)")
                continue
            m = Manifest.read(mpath)
            table.add_row(
                release,
                name,
                str(m.n_matched),
                str(m.groups.get("n_groups", "")),
                str(m.wall_s),
                m.created_at,
            )
    console.print(table)
