"""
Autoresearch dashboard — generates dashboard.html, auto-refreshes in browser.
Usage: uv run dashboard.py          # runs forever, updates every 15s
       uv run dashboard.py --once   # write once and exit
Then open dashboard.html in your browser.
"""

import os
import time
import argparse
import html
from pathlib import Path

import pandas as pd

REPO = Path(__file__).parent
RESULTS_TSV = REPO / "results.tsv"
RUN_LOG = REPO / "run.log"
OUTPUT_HTML = REPO / "dashboard.html"
REFRESH_INTERVAL = 15  # seconds
TAIL_LINES = 20


def load_results():
    if not RESULTS_TSV.exists():
        return None
    try:
        df = pd.read_csv(RESULTS_TSV, sep="\t")
        return df if not df.empty else None
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
    return bool(os.popen("pgrep -f 'train.py'").read().strip())


def build_html(df, log_lines, running):
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    status_color = "#22c55e" if running else "#94a3b8"
    status_text = "● TRAINING RUNNING" if running else "○ idle"

    # Summary stats
    summary_html = ""
    progress_html = ""
    if df is not None:
        keeps = df[df["status"] == "keep"]
        if not keeps.empty:
            best_bpb = keeps["val_bpb"].min()
            best_row = keeps.loc[keeps["val_bpb"].idxmin()]
            baseline_bpb = keeps.iloc[0]["val_bpb"]
            gain = baseline_bpb - best_bpb
            n_keep = len(keeps)
            n_discard = len(df[df["status"] == "discard"])
            n_crash = len(df[df["status"] == "crash"])
            summary_html = f"""
            <div class="summary-grid">
                <div class="card"><div class="label">Best val_bpb</div><div class="value best">{best_bpb:.6f}</div><div class="sub">{html.escape(str(best_row['commit']))}</div></div>
                <div class="card"><div class="label">Baseline</div><div class="value">{baseline_bpb:.6f}</div><div class="sub">improvement: -{gain:.6f}</div></div>
                <div class="card"><div class="label">Experiments</div><div class="value">{len(df)}</div><div class="sub">{n_keep} kept · {n_discard} discarded · {n_crash} crashed</div></div>
            </div>"""

            # Progress bar (improvement as % toward 0)
            pct = min(100, gain / baseline_bpb * 100 * 5)  # scaled for visibility
            progress_html = f"""
            <div class="progress-section">
                <div class="label">Progress from baseline ({baseline_bpb:.6f} → {best_bpb:.6f})</div>
                <div class="progress-bar"><div class="progress-fill" style="width:{pct:.1f}%"></div></div>
            </div>"""

    # Results table
    table_html = ""
    if df is not None:
        best_bpb_val = df[df["status"] == "keep"]["val_bpb"].min() if not df[df["status"] == "keep"].empty else None
        rows_html = ""
        for i, row in df.iterrows():
            bpb = f"{row['val_bpb']:.6f}" if row['val_bpb'] > 0 else "CRASH"
            status = str(row["status"])
            is_best = (status == "keep" and best_bpb_val is not None and row["val_bpb"] == best_bpb_val)
            row_class = f"row-{status}" + (" row-best" if is_best else "")
            star = "★ " if is_best else ""
            icon = {"keep": "✓", "discard": "✗", "crash": "☠"}.get(status, "?")
            desc = html.escape(str(row["description"]))
            rows_html += f"""
            <tr class="{row_class}">
                <td class="mono">{star}{html.escape(str(row['commit']))}</td>
                <td class="mono bpb">{bpb}</td>
                <td><span class="badge badge-{status}">{icon} {status}</span></td>
                <td>{desc}</td>
            </tr>"""
        table_html = f"""
        <table>
            <thead><tr><th>Commit</th><th>val_bpb</th><th>Status</th><th>Description</th></tr></thead>
            <tbody>{rows_html}</tbody>
        </table>"""

    # Log tail
    log_html = ""
    if log_lines:
        escaped = "\n".join(html.escape(l) for l in log_lines)
        log_html = f'<pre class="log">{escaped}</pre>'
    else:
        log_html = '<p class="dim">No run.log found.</p>'

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta http-equiv="refresh" content="{REFRESH_INTERVAL}">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>autoresearch dashboard</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif; background: #0f172a; color: #e2e8f0; padding: 24px; }}
  h1 {{ font-size: 1.4rem; font-weight: 600; color: #f8fafc; }}
  .header {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 24px; }}
  .status {{ font-size: 0.85rem; color: {status_color}; font-weight: 500; }}
  .meta {{ font-size: 0.75rem; color: #64748b; margin-top: 2px; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 16px; margin-bottom: 20px; }}
  .card {{ background: #1e293b; border-radius: 8px; padding: 16px; border: 1px solid #334155; }}
  .card .label {{ font-size: 0.75rem; color: #94a3b8; text-transform: uppercase; letter-spacing: 0.05em; margin-bottom: 6px; }}
  .card .value {{ font-size: 1.6rem; font-weight: 700; font-family: monospace; color: #f1f5f9; }}
  .card .value.best {{ color: #34d399; }}
  .card .sub {{ font-size: 0.75rem; color: #64748b; margin-top: 4px; font-family: monospace; }}
  .progress-section {{ margin-bottom: 24px; }}
  .progress-section .label {{ font-size: 0.8rem; color: #94a3b8; margin-bottom: 8px; }}
  .progress-bar {{ background: #1e293b; border-radius: 4px; height: 8px; overflow: hidden; }}
  .progress-fill {{ background: linear-gradient(90deg, #6366f1, #34d399); height: 100%; border-radius: 4px; transition: width 0.5s; }}
  h2 {{ font-size: 1rem; font-weight: 600; color: #cbd5e1; margin: 24px 0 12px; }}
  table {{ width: 100%; border-collapse: collapse; background: #1e293b; border-radius: 8px; overflow: hidden; border: 1px solid #334155; }}
  th {{ background: #0f172a; color: #94a3b8; font-size: 0.75rem; text-transform: uppercase; letter-spacing: 0.05em; padding: 10px 14px; text-align: left; }}
  td {{ padding: 10px 14px; font-size: 0.85rem; border-top: 1px solid #1e293b; vertical-align: top; }}
  .mono {{ font-family: monospace; }}
  .bpb {{ font-weight: 600; color: #e2e8f0; }}
  .row-keep td {{ background: #0f2420; }}
  .row-discard td {{ color: #64748b; }}
  .row-crash td {{ color: #ef4444; background: #1a0a0a; }}
  .row-best td {{ background: #0a2418 !important; }}
  .row-best .bpb {{ color: #34d399 !important; }}
  .badge {{ font-size: 0.75rem; padding: 2px 8px; border-radius: 9999px; font-weight: 500; }}
  .badge-keep {{ background: #14532d; color: #86efac; }}
  .badge-discard {{ background: #1e293b; color: #64748b; }}
  .badge-crash {{ background: #450a0a; color: #fca5a5; }}
  .log {{ background: #0f172a; border: 1px solid #334155; border-radius: 8px; padding: 16px; font-size: 0.78rem; font-family: monospace; color: #94a3b8; white-space: pre-wrap; word-break: break-all; line-height: 1.6; max-height: 400px; overflow-y: auto; }}
  .dim {{ color: #475569; font-size: 0.85rem; }}
</style>
</head>
<body>
<div class="header">
  <div>
    <h1>autoresearch dashboard</h1>
    <div class="meta">Updated {now} · auto-refreshes every {REFRESH_INTERVAL}s</div>
  </div>
  <div class="status">{status_text}</div>
</div>
{summary_html}
{progress_html}
<h2>Experiment History</h2>
{table_html if table_html else '<p class="dim">No results yet.</p>'}
<h2>Live Training Log</h2>
{log_html}
</body>
</html>"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--once", action="store_true", help="Write once and exit")
    parser.add_argument("--refresh", type=int, default=REFRESH_INTERVAL)
    args = parser.parse_args()

    while True:
        df = load_results()
        log_lines = tail_log()
        running = is_training_running()
        content = build_html(df, log_lines, running)
        OUTPUT_HTML.write_text(content)
        if args.once:
            print(f"Written to {OUTPUT_HTML}")
            break
        time.sleep(args.refresh)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        pass
