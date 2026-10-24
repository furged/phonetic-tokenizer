"""Per-group collision grid: each spelling group's token sequences under BPE and phonetic."""
import sys
from pathlib import Path

sys.path.insert(0, '.')

import csv
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from tokenizer.phonetic_g2p import build_phonetic_bpe
from baselines.bpe_tokenizer import BPETokenizer
from experiments.collision_rate import read_groups, collides

GROUPS_PATH = Path("data/raw/spelling_variants.csv")
TRAIN_PATH = "data/processed/train.txt"
OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

groups = read_groups(GROUPS_PATH)

phon = build_phonetic_bpe(TRAIN_PATH, vocab_size=2000)
bpe = BPETokenizer(vocab_size=2000)
bpe.train(TRAIN_PATH)

phon_enc = phon.encode
bpe_enc = bpe.encode

# Colliding = all variants identical; highlight the rows
rows = []
for g in groups:
    rows.append((g, collides(bpe_enc, g), collides(phon_enc, g)))

# Sort: phonetic-colliding groups first, then the rest
rows.sort(key=lambda r: (not r[2], not r[1]))

fig_h = max(4, 0.22 * len(rows) + 1.5)
fig, ax = plt.subplots(figsize=(10, fig_h))

grid = []
colors = []
labels = []
for g, b_hit, p_hit in rows:
    labels.append(" | ".join(g[:3]) + (" ..." if len(g) > 3 else ""))
    grid.append([int(b_hit), int(p_hit)])

import numpy as np
arr = np.array(grid)
cmap = matplotlib.colors.ListedColormap(["#E6E6E6", "#2E8B57"])
ax.imshow(arr, aspect="auto", cmap=cmap, vmin=0, vmax=1)

ax.set_xticks([0, 1])
ax.set_xticklabels(["BPE", "Phonetic"])
ax.set_yticks(range(len(rows)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_title("Per-group collisions: green = all spellings give identical tokens")
ax.legend(handles=[Patch(color="#2E8B57", label="collides"),
                   Patch(color="#E6E6E6", label="does not collide")],
          loc="upper left", bbox_to_anchor=(1.01, 1), fontsize=8)

fig.tight_layout()
fig.savefig(OUT / "collision_grid.png", dpi=150)
print(f"Saved {OUT / 'collision_grid.png'} ({len(rows)} groups)")