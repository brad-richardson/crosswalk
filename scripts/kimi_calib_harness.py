#!/usr/bin/env python3
"""Reusable Kimi OpenRouter provider calibration harness.

Isolated probe that replays archived evidence packs without touching
repo `opencode.json` or global `~/.config/opencode/opencode.jsonc`.

Uses highest-precedence OPENCODE_CONFIG_CONTENT env injection (SDK server.js
pattern) to override provider routing. Verifies precedence via
`opencode debug config`.

Expected panel behavior on 3 calibration packs:
- acfa90c9 small healthy N:1 -> settled A
- 8bf6c63b medium coverage-sensitive M:N -> settled A (unanimous)
  Option A=7 true edges, B-H=6 edges dropping one true edge. Q marks F=wrong.
- 99911d68 prior timeout case -> panel majority H (Kimi previously timed out)

Prior results (from 2026-07-12 investigation):
- p90:3 perf routing: fast (13-51s) but F on 8bf6c63b x2 -> FAIL quality
- Moonshot official only: timeout 240s on 8bf6c63b -> FAIL latency
- Baidu only: timeout 240s on 8bf6c63b -> FAIL latency
- WandB only: fast but F on 8bf6c63b -> FAIL quality
- Together only: fast but F on 8bf6c63b -> FAIL quality

21 endpoints exist. 16 remain untested for veto.

Budget: 30 min / 25 trials max per handoff objective.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).parent.parent
DEFAULT_BATCH = ROOT / "data/agents/stitching/batches/tn_tunis_ml_roads_sweep1_20260712"

SETTLED: dict[str, str] = {
    "acfa90c9": "A",
    "8bf6c63b": "A",
    "99911d68": "H",
}

# OpenRouter docs: provider object fields order, only, ignore, etc.
# We will test `only` and `order` plus quantization filters.

CANDIDATES_FROM_ENDPOINTS = [
    # fp8 — higher fidelity than int4/fp4, not yet tested
    ("siliconflow", "siliconflow/fp8", "fp8"),
    ("streamlake", "streamlake/fp8", "fp8"),
    # unknown quantization but distinct providers
    ("digitalocean", "digitalocean", "unknown"),
    ("novita", "novita", "unknown"),
    ("cloudflare", "cloudflare", "unknown"),
    ("parasail", "parasail/int4", "int4"),
    ("venice", "venice/int4", "int4"),
    ("inceptron", "inceptron/int4", "int4"),
    ("chutes", "chutes/int4", "int4"),
    ("deepinfra", "deepinfra/fp4", "fp4"),
    ("modelrun", "modelrun/fp4", "fp4"),
    ("nebius", "nebius/int4", "int4"),
    ("atlas-cloud", "atlas-cloud/int4", "int4"),
    ("baseten", "baseten/fp4", "fp4"),
    ("fireworks", "fireworks", "unknown"),
    ("phala", "phala", "unknown"),
]

# Already tried and failed:
FAILED_PROVIDERS = {"wandb", "together", "moonshotai", "baidu", "decart"}

_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> dict | None:
    if not text:
        return None
    m = _FENCE_RE.search(text)
    if m:
        try:
            d = json.loads(m.group(1).strip())
            if isinstance(d, dict):
                return d
        except Exception:
            pass
        # try scan
        idx = m.group(1).find("{")
        dec = json.JSONDecoder()
        while idx != -1:
            try:
                p, _ = dec.raw_decode(m.group(1), idx)
                if isinstance(p, dict):
                    return p
            except Exception:
                pass
            idx = m.group(1).find("{", idx + 1)

    dec = json.JSONDecoder()
    idx = text.find("{")
    while idx != -1:
        try:
            p, _ = dec.raw_decode(text, idx)
            if isinstance(p, dict):
                return p
        except Exception:
            pass
        idx = text.find("{", idx + 1)
    return None


def build_opencode_config(provider_policy: dict[str, Any]) -> dict[str, Any]:
    """Build a high-precedence config with model, vote agent, and routing.

    OPENCODE_CONFIG_CONTENT is merged over project/user configuration. Include
    every ballot-changing key so lower-precedence config cannot alter the
    selected model, endpoint policy, or tool-less vote agent.

    We copy vote agent definition from repo opencode.json and add openrouter
    model options.
    """
    repo_cfg = json.loads((ROOT / "opencode.json").read_text())
    vote_agent = repo_cfg.get("agent", {}).get("vote", {})

    return {
        "model": "openrouter/moonshotai/kimi-k2.6",
        "provider": {
            "openrouter": {
                "models": {"moonshotai/kimi-k2.6": {"options": {"provider": provider_policy}}}
            }
        },
        "agent": {"vote": vote_agent},
    }


def verify_isolation(provider_policy: dict[str, Any]) -> None:
    """Verify OPENCODE_CONFIG_CONTENT wins without writing live config.

    Run `opencode debug config` with our injected config and ensure
    openrouter provider shows our policy.
    """
    cfg = build_opencode_config(provider_policy)
    env = {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(cfg)}
    result = subprocess.run(
        ["opencode", "debug", "config"],
        capture_output=True,
        text=True,
        env=env,
        timeout=15,
    )
    if result.returncode != 0:
        raise RuntimeError(f"debug config failed: {result.stderr[:500]}")
    try:
        data = json.loads(result.stdout)
    except Exception as e:
        raise RuntimeError(f"debug config not JSON: {e} {result.stdout[:1000]}") from e
    prov = data.get("provider", {}).get("openrouter", {})
    models = prov.get("models", {}).get("moonshotai/kimi-k2.6", {})
    opts = models.get("options", {}).get("provider", {})
    if opts != provider_policy:
        raise RuntimeError(
            f"Isolation verification mismatch: expected {provider_policy} got {opts}"
        )
    print(f"[verify] Isolation OK: {provider_policy}")


def run_one_prompt(
    group_dir: Path,
    provider_policy: dict[str, Any],
    timeout_s: int = 240,
) -> dict[str, Any]:
    cfg = build_opencode_config(provider_policy)
    prompt = (group_dir / "prompt.txt").read_text()

    imgs = []
    ov = group_dir / "overview.png"
    if ov.exists():
        imgs.append(str(ov))
    for p in sorted(group_dir.glob("option_*.png")):
        imgs.append(str(p))
    for p in sorted(group_dir.glob("zoom_*.png")):
        imgs.append(str(p))

    cmd = ["opencode", "run", "-m", "openrouter/moonshotai/kimi-k2.6", "--agent", "vote"]
    for img in imgs:
        cmd += ["-f", img]

    env = {**os.environ, "OPENCODE_CONFIG_CONTENT": json.dumps(cfg)}
    # Each invocation its own DB to avoid sqlite locking when parallel waves,
    # same logic as stitch_runner.py
    db_dir = tempfile.mkdtemp(prefix="opencode_calib_db_")
    env["OPENCODE_DB"] = str(Path(db_dir) / "opencode.db")

    start = time.monotonic()
    try:
        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=timeout_s,
            env=env,
        )
        latency = time.monotonic() - start
        stdout = result.stdout or ""
        stderr = result.stderr or ""
        parsed = extract_json(stdout)
        choice = None
        confidence = None
        reasoning = None
        if parsed:
            choice = (
                str(parsed.get("choice", ""))
                .strip()
                .upper()
                .replace("OPTION", "")
                .strip()
                .strip(".")
            )
            confidence = parsed.get("confidence")
            reasoning = parsed.get("reasoning", "")[:500]
        return {
            "group_dir": str(group_dir),
            "provider_policy": provider_policy,
            "returncode": result.returncode,
            "latency_s": latency,
            "stdout": stdout[:4000],
            "stderr": stderr[:2000],
            "choice": choice,
            "confidence": confidence,
            "reasoning": reasoning,
            "parsed": parsed,
        }
    except subprocess.TimeoutExpired:
        latency = time.monotonic() - start
        return {
            "group_dir": str(group_dir),
            "provider_policy": provider_policy,
            "returncode": None,
            "latency_s": latency,
            "stdout": "",
            "stderr": "TIMEOUT",
            "choice": None,
            "confidence": None,
            "reasoning": "TIMEOUT",
            "parsed": None,
            "timeout": True,
        }
    finally:
        import shutil

        shutil.rmtree(db_dir, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser(description="Kimi calibration harness")
    ap.add_argument("--batch-root", type=Path, default=DEFAULT_BATCH)
    ap.add_argument("--groups", nargs="+", default=["8bf6c63b"])
    ap.add_argument("--provider", type=str, default="", help="single provider slug for only policy")
    ap.add_argument("--order", nargs="+", default=None, help="ordered provider slugs")
    ap.add_argument("--quantizations", nargs="+", default=None)
    ap.add_argument("--timeout", type=int, default=240)
    ap.add_argument("--verify", action="store_true")
    ap.add_argument("--list-candidates", action="store_true")
    ap.add_argument(
        "--full-sweep", action="store_true", help="sweep remaining candidates on veto pack"
    )
    args = ap.parse_args()

    if args.list_candidates:
        for name, tag, quant in CANDIDATES_FROM_ENDPOINTS:
            mark = "SKIP_FAILED" if name in FAILED_PROVIDERS else "CANDIDATE"
            print(f"{name:15s} {tag:20s} {quant:7s} {mark}")
        return

    if args.full_sweep:
        # Veto phase: test each remaining candidate ONLY on 8bf6c63b
        veto_group = args.batch_root / "8bf6c63b"
        results = []
        for name, tag, quant in CANDIDATES_FROM_ENDPOINTS:
            if name in FAILED_PROVIDERS:
                continue
            policy: dict[str, Any] = {"only": [name]}
            # for fp8 candidates, also allow fp8 quantization explicitly
            if quant in ("fp8", "unknown"):
                # no quant filter, just only
                pass
            print(f"\n=== VETO testing provider={name} tag={tag} on groups=8bf6c63b ===")
            if args.verify:
                verify_isolation(policy)
            res = run_one_prompt(veto_group, policy, timeout_s=args.timeout)
            choice = res.get("choice")
            settled = SETTLED["8bf6c63b"]
            ok = choice == settled
            print(
                f" -> choice={choice} expected={settled} ok={ok} latency={res['latency_s']:.1f}s rc={res['returncode']}"
            )
            if not ok:
                print(f"    reason={res['reasoning']!r} stderr={res['stderr'][:300]}")
            results.append(
                {
                    "provider": name,
                    "tag": tag,
                    "quantization": quant,
                    "policy": policy,
                    "result": res,
                    "passed_veto": ok,
                }
            )
        out = Path("/tmp/kimi_veto_results.json")
        out.write_text(json.dumps(results, indent=2))
        print(f"\nWrote {out}")
        passed = [r for r in results if r["passed_veto"]]
        print(f"Veto survivors: {len(passed)}/{len(results)}")
        for p in passed:
            print(
                f"  {p['provider']} {p['tag']} choice={p['result']['choice']} latency={p['result']['latency_s']:.1f}s"
            )
        return

    # Single run
    policies = []
    if args.provider:
        policies.append({"only": [args.provider]})
    if args.order:
        policies.append({"order": args.order, "allow_fallbacks": True})
    if args.quantizations:
        for q in args.quantizations:
            policies.append({"quantizations": [q]})
    if not policies:
        policies.append({})

    for group_id in args.groups:
        gdir = args.batch_root / group_id
        if not gdir.exists():
            print(f"Group {group_id} not found at {gdir}", file=sys.stderr)
            continue
        for policy in policies:
            print(f"\n--- group={group_id} policy={policy} ---")
            if args.verify:
                verify_isolation(policy)
            res = run_one_prompt(gdir, policy, timeout_s=args.timeout)
            settled = SETTLED.get(group_id, "?")
            ok = res.get("choice") == settled if settled != "?" else None
            print(
                f"choice={res.get('choice')} settled={settled} ok={ok} latency={res['latency_s']:.1f}s"
            )
            print(f"reasoning: {res.get('reasoning')}")
            if res.get("stderr"):
                print(f"stderr: {res['stderr'][:500]}")


if __name__ == "__main__":
    main()
