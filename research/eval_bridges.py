"""Evaluate already-computed bridges against labels using mbench eval functions.

Avoids re-running matcher for each (off/on) x (pair/target) combination.

Usage (from mbench/ venv):
    python <this> <dataset> <labels_human> <labels_stitch> <bridge1> [<bridge2> ...]
"""

import sys
from pathlib import Path

import pandas as pd
from mbench.adapters.matcher import MatcherAdapter
from mbench.eval.labels import load_labels, load_stitch_labels
from mbench.eval.metrics import evaluate
from mbench.eval.stitch_metrics import evaluate_stitch_groups


def main():
    dataset = sys.argv[1]
    labels_human = sys.argv[2]
    labels_stitch = sys.argv[3]
    bridges = sys.argv[4:]

    gt = load_labels(Path(labels_human), dataset)
    stitch_labels = load_stitch_labels(Path(labels_stitch), dataset)
    adapter = MatcherAdapter()

    rows = []
    for bridge in bridges:
        matches = adapter.parse_output(bridge).matches
        target_eval = evaluate(matches, gt, match_level="target")
        pair_eval = evaluate(matches, gt, match_level="pair")
        row = {"bridge": bridge.split("/")[-1]}
        for lvl, ev in [("target", target_eval), ("pair", pair_eval)]:
            row[f"{lvl}_P"] = round(ev.precision, 4)
            row[f"{lvl}_R"] = round(ev.recall, 4)
            row[f"{lvl}_F1"] = round(ev.f1, 4)
            row[f"{lvl}_TP"] = ev.true_positives
            row[f"{lvl}_FP"] = ev.false_positives
            row[f"{lvl}_FN"] = ev.false_negatives
        if stitch_labels is not None:
            st = evaluate_stitch_groups(matches, stitch_labels)
            row["stitch_P"] = round(st.precision, 4)
            row["stitch_R"] = round(st.recall, 4)
            row["stitch_F1"] = round(st.f1, 4)
            row["stitch_groups"] = st.groups_evaluated
        rows.append(row)

    df = pd.DataFrame(rows)
    pd.set_option("display.width", 200)
    pd.set_option("display.max_columns", 50)
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
