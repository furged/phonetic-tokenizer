"""Vocab size chart: base alphabet vs final vocab for BPE and phonetic."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

# Base alphabet: unique characters in the 50k training text (measured)
chars = set()
with open("data/processed/train.txt", encoding="utf-8") as f:
    for line in f:
        chars.update(line.strip())
base_bpe = len(chars)
base_phon = 51  # IPA symbols from the eSpeak run

labels = ["BPE", "Phonetic-BPE"]
base = [base_bpe, base_phon]
final = [2000, 2000]

x = np.arange(len(labels))
w = 0.35
fig, ax = plt.subplots(figsize=(6.5, 4.5))
b1 = ax.bar(x - w / 2, base, w, label="base symbols (start)", color="#AAAAAA")
b2 = ax.bar(x + w / 2, final, w, label="final vocab", color=["#4C72B0", "#DD8452"])
for bars in (b1, b2):
    for b in bars:
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 25, f"{int(b.get_height())}",
                ha="center", fontsize=9)
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.set_ylabel("number of symbols")
ax.set_title("Starting alphabet vs final vocabulary (2000 each)")
ax.legend()
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUT / "vocab_compression.png", dpi=150)
print(f"Saved {OUT / 'vocab_compression.png'} (BPE base chars: {base_bpe})")