"""Collision rate: do spelling variants of the same word map to the same tokens?"""

import sys
sys.path.insert(0, '.')

import csv
from pathlib import Path
from tokenizer.phonetic_g2p import PhoneticTokenizer, build_phonetic_bpe
from baselines.bpe_tokenizer import BPETokenizer

GROUPS_PATH = Path("data/raw/spelling_variants.csv")
TRAIN_PATH = "data/processed/train.txt"
OUT_PATH = Path("results/tables/collision_rate.csv")


def read_groups(path):
    """Each row is one word's spelling variants. Empty cells are skipped."""
    groups = []
    with open(path, encoding="utf-8") as f:
        for row in csv.reader(f):
            variants = [v.strip() for v in row if v.strip()]
            if len(variants) >= 2:
                groups.append(variants)
    return groups


def collides(encode, variants):
    """True if every variant in the group gets the same token sequence."""
    seqs = [tuple(encode(v)) for v in variants]
    return len(set(seqs)) == 1


def main():
    groups = read_groups(GROUPS_PATH)
    print(f"Loaded {len(groups)} spelling groups")

    phonetic_bpe = build_phonetic_bpe(TRAIN_PATH, vocab_size=2000)
    bpe = BPETokenizer(vocab_size=2000)
    bpe.train(TRAIN_PATH)

    # Wrap both so they expose encode() -> list of ids
    phon_encode = lambda s: phonetic_bpe.encode(s).ids
    bpe_encode = bpe.encode

    phon_hits = sum(collides(phon_encode, g) for g in groups)
    bpe_hits = sum(collides(bpe_encode, g) for g in groups)
    n = len(groups)

    print(f"BPE collision rate:      {bpe_hits}/{n} = {bpe_hits/n:.3f}")
    print(f"Phonetic collision rate: {phon_hits}/{n} = {phon_hits/n:.3f}")

    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["Tokenizer", "Groups", "Colliding groups", "Collision rate"])
        w.writerow(["BPE", n, bpe_hits, f"{bpe_hits/n:.3f}"])
        w.writerow(["Phonetic", n, phon_hits, f"{phon_hits/n:.3f}"])
    print(f"Saved to {OUT_PATH}")


if __name__ == "__main__":
    main()