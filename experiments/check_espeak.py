"""Check eSpeak G2P on all spelling variants.

For each group, the 'reference' is the most frequent IPA output among its variants.
A variant is 'correct' if its IPA matches the reference. Reports per-group and overall accuracy.
"""
import sys
sys.path.insert(0, '.')

import csv
from collections import Counter
from pathlib import Path

from tokenizer.phonetic_g2p import espeak_phonemes

GROUPS_PATH = Path("data/raw/spelling_variants.csv")
OUT = Path("results/tables/espeak_check.csv")


def read_groups(path):
    groups = []
    with open(path, encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        variant_cols = [i for i, h in enumerate(header) if h.startswith("variant_")]
        for row in reader:
            variants = [row[i].strip() for i in variant_cols if i < len(row) and row[i].strip()]
            if len(variants) >= 2:
                hindi = row[0].strip()
                groups.append((hindi, variants))
    return groups


def main():
    groups = read_groups(GROUPS_PATH)
    total = 0
    correct = 0
    per_group = []

    for hindi, variants in groups:
        ipa = {v: "".join(espeak_phonemes(v)) for v in variants}
        ref, _ = Counter(ipa.values()).most_common(1)[0]
        n_ok = sum(1 for v in variants if ipa[v] == ref)
        total += len(variants)
        correct += n_ok
        per_group.append((hindi, " | ".join(variants), n_ok, len(variants), ref))

    acc = correct / total if total else 0.0
    print(f"Groups: {len(groups)}, variants: {total}")
    print(f"eSpeak agreement with group reference: {correct}/{total} = {acc:.3f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["hindi", "variants", "agree", "total", "reference_ipa"])
        for r in per_group:
            w.writerow(r)
    print(f"Saved {OUT}")


if __name__ == "__main__":
    main()