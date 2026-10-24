"""Sound mapping table: spellings of one word, their eSpeak phonemes, and collision status."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = Path("results/plots")
OUT.mkdir(parents=True, exist_ok=True)

# Values from the eSpeak test run (Phonemizer, hi)
rows = [
    ["acha",   "ɐ t ʃ a",   "ɐ t ʃ a"],
    ["accha",  "ɐ k h a",   "ɐ k h a"],
    ["achha",  "ɐ t ʃ h a", "ɐ t ʃ h a"],
]
cols = ["Spelling", "Phonetic (eSpeak IPA)", "Note"]
rows[0].append("reference")
rows[1].append("reads cc as k")
rows[2].append("differs from acha by h")
cols = ["Spelling", "eSpeak IPA", "Note"]
rows = [[r[0], r[1], r[3]] for r in rows]

fig, ax = plt.subplots(figsize=(7, 2.6))
ax.axis("off")
table = ax.table(cellText=rows, colLabels=cols, loc="center", cellLoc="left")
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1, 1.8)
ax.set_title("Same word, different spelling: how eSpeak reads each variant", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "sound_mapping.png", dpi=150)
print("Saved", OUT / "sound_mapping.png")