"""Test Latin -> Devanagari -> Epitran IPA on spelling variant groups."""
import sys
sys.path.insert(0, '.')

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import epitran

from experiments.collision_rate import read_groups

epi = epitran.Epitran('hin-Deva')
groups = read_groups("data/raw/spelling_variants.csv")

for variants in groups[:10]:
    print("GROUP:", " | ".join(variants))
    for w in variants:
        deva = transliterate(w, sanscript.ITRANS, sanscript.DEVANAGARI)
        ipa = epi.transliterate(deva)
        print(f"  {w:<12} -> {deva:<12} -> {ipa}")
    print()