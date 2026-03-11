"""
Live dashboard for autoresearch experiments.
Usage: uv run dashboard.py [--refresh 10]
Shows results history and live training progress.
"""

import os
import sys
import time
import argparse
import pandas as pd
from pathlib import Path

REPO = Path(__file__).parent
RESULTS_TSV = REPO / "results.tsv"
RUN_LOG = REPO / "run.log"
TAIL_LINES = 8


def clear():
    os.system("clear")


def load_results():
    if not RESULTS_TSV.exists():
        return None
    try:
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        return df
    except Exception:
        return None


def tail_log(n=TAIL_LINES):
    if not RUN_LOG.exists():
        return []
    try:
        with open(RUN_LOG) as f:
            lines = f.readlines()
        return [l.rstrip() for l in lines[-n:]]
    except Exception:
        return []


def is_training_running():
    """Check if train.py is currently running."""
    ret = os.popen("pgrep -f 'train.py'").read().strip()
    return bool(ret)


def render(df, log_lines, running):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    status = "TRAINING RUNNING" if running else "idle"
    print(f"╔══ autoresearch dashboard ══ {now} ══ {status} ══╗")
    print()

    if df is None or df.empty:
        print("  No results yet.")
    else:
        keeps = df[df["status"] == "keep"]
        if not keeps.empty:
            best = keeps["val_bpb"].min()
            best_row = keeps.loc[keeps["val_bpb"].idxmin()]
            print(f"  Best val_bpb : {best:.6f}  [{best_row['commit']}]  {best_row['description']}")
            baseline_row = keeps.iloc[0]
            if len(keeps) > 1:
                gain = baseline_row["val_bpb"] - best
                print(f"  vs baseline  : {baseline_row['val_bpb']:.6f}  (improvement: -{gain:.6f})")
        print()

        # Results table
        print(f"  {'#':<4} {'commit':<9} {'val_bpb':<12} {'mem_gb':<8} {'status':<9} description")
        print(f"  {'─'*4} {'─'*9} {'─'*12} {'─'*8} {'─'*9} {'─'*40}")
        for i, row in df.iterrows():
            bpb = f"{row['val_bpb']:.6f}" if row['val_bpb'] > 0 else "CRASH"
            mem = f"{row['memory_gb']:.1f}" if row['memory_gb'] > 0 else "—"
            status_sym = {"keep": "✓", "discard": "✗", "crash": "☠"}.get(row["status"], "?")
            desc = str(row["description"])[:55]
            is_best = (row["status"] == "keep" and row["val_bpb"] == df[df["status"]=="keep"]["val_bpb"].min())
            marker = " ★" if is_best else "  "
            print(f"{marker} {i+1:<3} {row['commit']:<9} {bpb:<12} {mem:<8} {status_sym:<9} {desc}")

        # Summary stats
        n_keep = len(df[df["status"] == "keep"])
        n_discard = len(df[df["status"] == "discard"])
        n_crash = len(df[df["status"] == "crash"])
        print()
        print(f"  Experiments: {len(df)} total | {n_keep} kept | {n_discard} discarded | {n_crash} crashes")

    # Live log tail
    print()
    print(f"  ── run.log (last {TAIL_LINES} lines) ──────────────────────────────")
    if log_lines:
        for line in log_lines:
            print(f"  {line[:100]}")
    else:
        print("  (no run.log)")
    print()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--refresh", type=int, default=10, help="Refresh interval in seconds")
    parser.add_argument("--once", action="store_true", help="Print once and exit")
    args = parser.parse_args()

    while True:
        df = load_results()
        log_lines = tail_log()
        running = is_training_running()
        clear()
        render(df, log_lines, running)
        if args.once:
            break
        print(f"  Refreshing every {args.refresh}s — Ctrl+C to quit")
        time.sleep(args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nDone.")
