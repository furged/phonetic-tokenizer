"""Per-epoch training loss, both tokenizers, from recorded values only."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Values as recorded from training output
bpe = {1: 4.16, 2: 3.98, 3: 3.86, 4: 3.79, 5: 3.78}
phon = {1: 4.91, 2: 4.56, 3: 4.38, 4: 4.12, 5: 4.04}
bpe_final = 3.77
phon_final = 4.03

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(7.5, 4.8))
for data, label, color, final in [
    (bpe, "BPE", "#4C72B0", bpe_final),
    (phon, "Phonetic-BPE", "#DD8452", phon_final),
]:
    ep = sorted(data)
    ax.plot(ep, [data[e] for e in ep], "-o", color=color, linewidth=2, label=label)
    for e in ep:
        ax.annotate(f"{data[e]:.2f}", (e, data[e]), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=8, color=color)
    ax.axhline(final, color=color, linestyle=":", linewidth=1,
               label=f"{label} final: {final:.2f}")

ax.set_xlabel("epoch")
ax.set_ylabel("train loss (nats/token)")
ax.set_title("Training loss per epoch (recorded values)")
ax.set_xticks([1, 2, 3, 4, 5])
ax.legend(fontsize=8)
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "loss_per_epoch.png", dpi=150)
print("Saved", OUT / "loss_per_epoch.png")