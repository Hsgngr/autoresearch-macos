"""Simple progress dashboard. Usage: uv run plot_progress.py"""
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe

df = pd.read_csv("results.tsv", sep="\t")
df = df[df["status"].isin(["keep", "discard", "crash"])]
df = df.reset_index(drop=True)
df.index.name = "exp"

kept = df[df["status"] == "keep"]
discarded = df[df["status"] != "keep"]

# Running best line (step function through kept experiments)
best_x, best_y = [], []
current_best = float("inf")
for i, row in df.iterrows():
    if row["status"] == "keep" and row["val_bpb"] < current_best:
        current_best = row["val_bpb"]
    best_x.append(i)
    best_y.append(current_best if current_best < float("inf") else None)

fig, ax = plt.subplots(figsize=(12, 6))

ax.scatter(discarded.index, discarded["val_bpb"], color="#cccccc", s=20, zorder=2, label="Discarded")
ax.scatter(kept.index, kept["val_bpb"], color="#2ecc71", s=40, zorder=3, label="Kept")
ax.step(best_x, best_y, where="post", color="#27ae60", linewidth=1.5, zorder=1, label="Running best")

# Label kept experiments
for i, row in kept.iterrows():
    desc = row["description"][:30] if len(row["description"]) > 30 else row["description"]
    ax.annotate(desc, (i, row["val_bpb"]),
                textcoords="offset points", xytext=(5, -8),
                fontsize=6, color="#27ae60", rotation=30,
                path_effects=[pe.withStroke(linewidth=2, foreground="white")])

n_total = len(df)
n_kept = len(kept)
ax.set_title(f"Autoresearch Progress: {n_total} Experiments, {n_kept} Kept Improvements")
ax.set_xlabel("Experiment #")
ax.set_ylabel("Validation BPB (lower is better)")
ax.legend(loc="upper right", fontsize=8)
ax.set_xlim(left=-0.5, right=max(len(df) - 0.5, 4))
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig("progress.png", dpi=150)
print(f"Saved progress.png ({n_total} experiments, best val_bpb: {kept['val_bpb'].min():.6f}" if n_kept else f"Saved progress.png ({n_total} experiments, no kept yet)")
