"""Hootenanny Docker integration for conflation benchmarking.

Supports two modes:
1. docker-compose mode (recommended): Uses official Hootenanny docker-compose setup
2. standalone mode: Uses custom Dockerfile (requires building from source)
"""

import os
import subprocess
from pathlib import Path

# Default to docker-compose mode
HOOTENANNY_MODE = os.environ.get("HOOTENANNY_MODE", "compose")
HOOTENANNY_DIR = os.environ.get("HOOTENANNY_DIR", None)


def _get_hoot_dir() -> Path:
    """Get path to Hootenanny repo for docker-compose mode."""
    if HOOTENANNY_DIR:
        return Path(HOOTENANNY_DIR)
    # Default: sibling directory to matcher
    return Path(__file__).parents[4] / "hootenanny"


def is_compose_running() -> bool:
    """Check if Hootenanny docker-compose services are running."""
    hoot_dir = _get_hoot_dir()
    if not hoot_dir.exists():
        return False
    try:
        result = subprocess.run(
            ["docker", "compose", "ps", "-q", "core-services"],
            cwd=hoot_dir,
            capture_output=True,
            text=True,
        )
        return bool(result.stdout.strip())
    except Exception:
        return False


def ensure_compose_running() -> None:
    """Start Hootenanny docker-compose if not already running."""
    hoot_dir = _get_hoot_dir()
    if not hoot_dir.exists():
        raise FileNotFoundError(
            f"Hootenanny repo not found at {hoot_dir}. "
            "Clone it with: git clone https://github.com/ngageoint/hootenanny.git"
        )
    if not is_compose_running():
        print("Starting Hootenanny services...")
        subprocess.run(
            ["make", "-f", "Makefile.docker", "up"],
            cwd=hoot_dir,
            check=True,
        )


def run_hoot(*args, data_dir: Path) -> subprocess.CompletedProcess:
    """Run a hoot command via Docker.

    Args:
        *args: Arguments to pass to hoot command
        data_dir: Data directory to use for file paths

    Returns:
        CompletedProcess with stdout/stderr
    """
    if HOOTENANNY_MODE == "compose":
        return _run_hoot_compose(*args, data_dir=data_dir)
    else:
        return _run_hoot_standalone(*args, data_dir=data_dir)


def _run_hoot_compose(*args, data_dir: Path) -> subprocess.CompletedProcess:
    """Run hoot via docker-compose exec."""
    ensure_compose_running()
    hoot_dir = _get_hoot_dir()

    # Convert paths: data_dir files need to be accessible in container
    # The container mounts the hootenanny dir, so we copy data there
    container_data_dir = hoot_dir / "data"
    container_data_dir.mkdir(exist_ok=True)

    # Rewrite any /data/ paths to use the container's data directory
    rewritten_args = []
    for arg in args:
        if isinstance(arg, str) and arg.startswith("/data/"):
            # Map /data/foo to the hootenanny/data/foo path
            rel_path = arg[6:]  # Remove /data/
            rewritten_args.append(f"/var/lib/hootenanny/data/{rel_path}")
        else:
            rewritten_args.append(str(arg))

    cmd = [
        "docker",
        "compose",
        "exec",
        "-T",  # Disable TTY for scripting
        "core-services",
        "/var/lib/hootenanny/bin/hoot",
        *rewritten_args,
    ]
    return subprocess.run(cmd, cwd=hoot_dir, capture_output=True, text=True, check=True)


def _run_hoot_standalone(*args, data_dir: Path) -> subprocess.CompletedProcess:
    """Run hoot via standalone Docker image."""
    docker_image = "hootenanny-cli:latest"
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{data_dir}:/data",
        docker_image,
        "hoot",
        *args,
    ]
    return subprocess.run(cmd, capture_output=True, text=True, check=True)


def copy_to_hoot_data(src: Path, dest_name: str) -> Path:
    """Copy a file to Hootenanny's data directory for compose mode.

    Args:
        src: Source file path
        dest_name: Destination filename within hootenanny/data/

    Returns:
        Path that can be used in hoot commands (container path)
    """
    import shutil

    hoot_dir = _get_hoot_dir()
    data_dir = hoot_dir / "data"
    data_dir.mkdir(exist_ok=True)
    dest = data_dir / dest_name
    shutil.copy2(src, dest)
    return Path(f"/var/lib/hootenanny/data/{dest_name}")


def conflate(
    reference: Path,
    target: Path,
    output: Path,
    data_dir: Path,
    match_creators: str = "HighwayMatchCreator",
    merger_creators: str = "HighwayMergerCreator",
) -> Path:
    """Run Hootenanny conflation on two OSM files.

    Args:
        reference: Path to reference OSM file (relative to data_dir)
        target: Path to target OSM file (relative to data_dir)
        output: Path for output OSM file (relative to data_dir)
        data_dir: Data directory containing input files
        match_creators: Hootenanny match creator (default: roads only)
        merger_creators: Hootenanny merger creator (default: roads only)

    Returns:
        Path to output file
    """
    if HOOTENANNY_MODE == "compose":
        # Copy files to hootenanny data dir and run
        ref_container = copy_to_hoot_data(data_dir / reference, reference.name)
        tgt_container = copy_to_hoot_data(data_dir / target, target.name)
        out_container = Path(f"/var/lib/hootenanny/data/{output.name}")

        run_hoot(
            "conflate",
            "-D",
            f"match.creators={match_creators}",
            "-D",
            f"merger.creators={merger_creators}",
            str(ref_container),
            str(tgt_container),
            str(out_container),
            data_dir=data_dir,
        )

        # Copy result back
        hoot_dir = _get_hoot_dir()
        result_path = hoot_dir / "data" / output.name
        import shutil

        final_output = data_dir / output
        final_output.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(result_path, final_output)
        return final_output
    else:
        run_hoot(
            "conflate",
            "-D",
            f"match.creators={match_creators}",
            "-D",
            f"merger.creators={merger_creators}",
            f"/data/{reference}",
            f"/data/{target}",
            f"/data/{output}",
            data_dir=data_dir,
        )
        return data_dir / output
