"""Print CLAUDE.md-style stats for a bridge parquet.

Usage: python research/bridge_stats.py <bridge.parquet> [run.log]

Reports Matched / Review / Unmatched (from the run log if given) and edge/group
counts derived from the bridge itself.
"""

import re
import sys
from pathlib import Path

import pandas as pd


def main():
    bridge_path = Path(sys.argv[1])
    log_path = Path(sys.argv[2]) if len(sys.argv) > 2 else None

    b = pd.read_parquet(bridge_path)
    mt = b["match_type"].value_counts().to_dict() if "match_type" in b else {}

    n_1n_groups = b.loc[b["match_type"] == "1:N", "gers_id"].nunique() if "match_type" in b else 0
    n_n1_groups = b.loc[b["match_type"] == "N:1", "local_id"].nunique() if "match_type" in b else 0

    matched = review = unmatched = None
    if log_path and log_path.exists():
        text = log_path.read_text()
        for key in ("Matched", "Review", "Unmatched"):
            m = re.search(rf"{key}:\s*(-?\d+)\s*$", text, re.MULTILINE)
            if m:
                val = int(m.group(1))
                if key == "Matched":
                    matched = val
                elif key == "Review":
                    review = val
                else:
                    unmatched = val

    print(f"=== {bridge_path.name} ===")
    print(f"  Total bridge edges : {len(b)}")
    print(f"  Matched (targets)  : {matched}")
    print(f"  Review (targets)   : {review}")
    print(f"  Unmatched (targets): {unmatched}")
    print(f"  1:1 edges          : {mt.get('1:1', 0)}")
    print(f"  1:N edges          : {mt.get('1:N', 0)}  (groups: {n_1n_groups})")
    print(f"  N:1 edges          : {mt.get('N:1', 0)}  (groups: {n_n1_groups})")
    print(f"  M:N edges          : {mt.get('M:N', 0)}")
    print(f"  mean confidence    : {b['confidence'].mean():.4f}")


if __name__ == "__main__":
    main()
