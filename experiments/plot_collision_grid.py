"""Graded per-group grid: number of distinct token sequences per spelling group."""
import sys
from pathlib import Path
import random

sys.path.insert(0, '.')

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from tokenizer.phonetic_g2p import build_phonetic_bpe
from baselines.bpe_tokenizer import BPETokenizer
from experiments.collision_rate import read_groups

GROUPS_PATH = Path("data/raw/spelling_variants.csv")
TRAIN_PATH = "data/processed/train.txt"
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

groups = read_groups(GROUPS_PATH)

phon = build_phonetic_bpe(TRAIN_PATH, vocab_size=2000)
bpe = BPETokenizer(vocab_size=2000)
bpe.train(TRAIN_PATH)

def distinct(encode, variants):
    return len({tuple(encode(v)) for v in variants})

rows = []
for g in groups:
    rows.append((g, distinct(bpe.encode, g), distinct(phon.encode, g)))

random.Random(42).shuffle(rows)

grid = np.array([[r[1], r[2]] for r in rows], dtype=float)
labels = [" | ".join(r[0][:3]) + (" ..." if len(r[0]) > 3 else "") for r in rows]

fig_h = max(5, 0.2 * len(rows) + 1.5)
fig, ax = plt.subplots(figsize=(7, fig_h))
im = ax.imshow(grid, aspect="auto", cmap="viridis_r", vmin=1, vmax=max(2, grid.max()))

ax.set_xticks([0, 1])
ax.set_xticklabels(["BPE", "Phonetic"])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(labels, fontsize=6)
for i in range(grid.shape[0]):
    for j in range(grid.shape[1]):
        ax.text(j, i, int(grid[i, j]), ha="center", va="center", fontsize=6,
                color="white" if grid[i, j] > 1 else "black")

cbar = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.04)
cbar.set_label("distinct token sequences (1 = all spellings identical)")
ax.set_title("Per-group spread: fewer distinct sequences = more collapse")
fig.tight_layout()
fig.savefig(OUT / "collision_grid.png", dpi=150)
print(f"Saved {OUT / 'collision_grid.png'} ({len(rows)} groups)")
