"""Bar chart of test bits per character, from bpc.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/tables/bpc.csv")
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

rows = [r for r in csv.reader(open(SRC, encoding="utf-8")) if len(r) == 2 and r[0] in ("bpe", "phonetic")]
names = ["BPE" if r[0] == "bpe" else "Phonetic" for r in rows]
vals = [float(r[1]) for r in rows]

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(names, vals, color=["#4C72B0", "#DD8452"][:len(vals)], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 0.02, f"{v:.4f}", ha="center", fontsize=10)
ax.set_ylabel("bits per character (lower is better)")
ax.set_title("Test bits per character, same held-out text")
ax.set_ylim(0, max(vals) * 1.2)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "bpc.png", dpi=150)
print("Saved", OUT / "bpc.png")