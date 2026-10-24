"""Read local L3Cube-HingCorpus archive and write a fixed seeded 50k subset (reservoir sampling)."""
import random
import tarfile
from pathlib import Path

SEED = 42
N = 50000
ARCHIVE = Path("L3Cube-HingCorpus_roman.tar.gz")
OUT = Path("data/processed/train.txt")

rng = random.Random(SEED)
reservoir = []
seen = 0

with tarfile.open(ARCHIVE, "r:gz") as tar:
    for member in tar:
        if not member.isfile():
            continue
        f = tar.extractfile(member)
        for raw in f:
            text = raw.decode("utf-8", errors="ignore").strip()
            if not text:
                continue
            seen += 1
            if len(reservoir) < N:
                reservoir.append(text)
            else:
                j = rng.randint(0, seen - 1)
                if j < N:
                    reservoir[j] = text
        print(f"Done member: {member.name}, lines seen so far: {seen}")

print(f"Total non-empty lines: {seen}")
OUT.parent.mkdir(parents=True, exist_ok=True)
with open(OUT, "w", encoding="utf-8") as f:
    for s in reservoir:
        f.write(s + "\n")
print(f"Wrote {len(reservoir)} sentences to {OUT}")