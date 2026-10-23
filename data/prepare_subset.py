"""Stream L3Cube-HingCorpus and write a fixed seeded subset."""
import random
from datasets import load_dataset
from pathlib import Path

SEED = 42
N = 50000
OUT = Path("data/processed/train.txt")

ds = load_dataset("hinglish-orthobench/HingCorpus", split="train", streaming=True)
ds = ds.shuffle(seed=SEED, buffer_size=10000)

OUT.parent.mkdir(parents=True, exist_ok=True)
written = 0
with open(OUT, "w", encoding="utf-8") as f:
    for ex in ds:
        text = ex.get("text") or next(iter(ex.values()))
        text = str(text).strip().replace("\n", " ")
        if text:
            f.write(text + "\n")
            written += 1
        if written >= N:
            break
print(f"Wrote {written} sentences to {OUT}")