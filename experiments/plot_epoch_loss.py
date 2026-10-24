"""Per-epoch training loss chart from epoch-end values."""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Usage: python experiments/plot_epoch_loss.py phonetic 4.10 4.08 4.06 4.05 4.04
NAME = sys.argv[1]
losses = [float(x) for x in sys.argv[2:]]
if not losses:
    sys.exit("Pass epoch-end losses after the name, e.g. phonetic 4.10 4.08 ...")

epochs = list(range(1, len(losses) + 1))
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7, 4.5))
ax.plot(epochs, losses, marker="o", color="#DD8452", linewidth=2)
for e, l in zip(epochs, losses):
    ax.annotate(f"{l:.3f}", (e, l), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=9)
ax.set_xlabel("epoch")
ax.set_ylabel("train loss (nats/token)")
ax.set_title(f"Per-epoch training loss: {NAME}")
ax.set_xticks(epochs)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / f"epoch_loss_{NAME}.png", dpi=150)
print(f"Saved {OUT / f'epoch_loss_{NAME}.png'}")