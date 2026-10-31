"""Collision rate on the Latin -> Devanagari -> Epitran route, all groups."""
import sys
sys.path.insert(0, '.')

from indic_transliteration import sanscript
from indic_transliteration.sanscript import transliterate
import epitran

from experiments.collision_rate import read_groups, collides

epi = epitran.Epitran('hin-Deva')

def epitran_encode(word):
    # Same route as the test script: Latin -> Devanagari -> IPA symbols
    deva = transliterate(word, sanscript.ITRANS, sanscript.DEVANAGARI)
    ipa = epi.transliterate(deva)
    return list(ipa.replace(' ', ''))

groups = read_groups("data/raw/spelling_variants.csv")
hits = [g for g in groups if collides(epitran_encode, g)]

print(f"Groups: {len(groups)}")
print(f"Epitran route collisions: {len(hits)}/{len(groups)} = {len(hits)/len(groups):.3f}")
for g in hits:
    print("  collides:", " | ".join(g))