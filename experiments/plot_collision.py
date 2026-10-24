"""Bar chart of collision rate for BPE vs phonetic."""
from pathlib import Path
import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/tables/collision_rate.csv")
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

rows = list(csv.DictReader(open(SRC, encoding="utf-8")))
names = [r["Tokenizer"] for r in rows]
rates = [float(r["Collision rate"]) for r in rows]
hits = [int(r["Colliding groups"]) for r in rows]
n = int(rows[0]["Groups"])

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(names, rates, color=["#4C72B0", "#DD8452"], width=0.5)
for bar, h, r in zip(bars, hits, rates):
    ax.text(bar.get_x() + bar.get_width() / 2, r + 0.005,
            f"{h}/{n}\n({r:.1%})", ha="center", va="bottom", fontsize=10)

ax.set_ylabel("Collision rate (share of spelling groups)")
ax.set_title("Do spelling variants map to the same tokens?")
ax.set_ylim(0, max(rates) * 1.3 + 0.02)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "collision_rate.png", dpi=150)
print(f"Saved {OUT / 'collision_rate.png'}")