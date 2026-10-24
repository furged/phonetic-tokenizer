"""Bar chart of average tokens per sentence, from tokenizer_stats.csv."""
import csv
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = Path("results/tables/tokenizer_stats.csv")
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

rows = list(csv.reader(open(SRC, encoding="utf-8")))[1:]
names = [r[0] for r in rows if r[0] in ("BPE (2000)", "Phonetic-BPE (2000)")]
vals = [float(r[2]) for r in rows if r[0] in ("BPE (2000)", "Phonetic-BPE (2000)")]

fig, ax = plt.subplots(figsize=(6, 4.5))
bars = ax.bar(names, vals, color=["#4C72B0", "#DD8452"], width=0.5)
for b, v in zip(bars, vals):
    ax.text(b.get_x() + b.get_width() / 2, v + 1, f"{v:.1f}", ha="center", fontsize=10)
ax.set_ylabel("avg tokens per sentence")
ax.set_title("Sequence length: 50k sentence subset")
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "tokens_per_sentence.png", dpi=150)
print("Saved", OUT / "tokens_per_sentence.png")