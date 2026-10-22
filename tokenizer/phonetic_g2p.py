"""Phonetic tokenizer for Hinglish using rule-based phoneme conversion + BPE merging."""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer
import re

from phonemizer import phonemize

def espeak_phonemes(word):
    """Latin Hinglish word -> list of IPA phoneme symbols via eSpeak."""
    out = phonemize(word, language='hi', backend='espeak', strip=True)
    out = re.sub(r'\((en|hi)\)', '', out).strip()
    return list(out.replace(' ', ''))

class PhoneticTokenizer:
    """Rule-based phonetic tokenizer for romanized Hinglish."""
    
    def __init__(self):
        self.phoneme_to_id = {}
        self.id_to_phoneme = {}
        self.vocab_built = False
    
    def text_to_phonemes_rules(self, text):
        """OLD baseline: hand-written digraph rules (kept for comparison)."""
        text = text.lower().strip()
        phonemes = []
        i = 0
        while i < len(text):
            if text[i].isspace():
                i += 1
                continue
            if i + 1 < len(text) and text[i:i+2] in ['ch', 'sh', 'th', 'ph', 'kh', 'gh', 'dh', 'bh', 'ng']:
                phonemes.append(text[i:i+2])
                i += 2
                continue
            phonemes.append(text[i])
            i += 1
        return phonemes

    def text_to_phonemes(self, text):
        """Hinglish text -> phoneme list using eSpeak (Hindi). This is the main method."""
        phonemes = []
        for word in text.lower().split():
            phonemes.extend(espeak_phonemes(word))
        return phonemes
    
    def build_vocab(self, corpus_path, min_freq=1):
        """Build phoneme vocabulary from corpus."""
        from collections import Counter
        phoneme_counts = Counter()
        
        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in f:
                phonemes = self.text_to_phonemes(line.strip())
                phoneme_counts.update(phonemes)
        
        special_tokens = ['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        idx = 0
        for token in special_tokens:
            self.phoneme_to_id[token] = idx
            self.id_to_phoneme[idx] = token
            idx += 1
        
        for phoneme, count in phoneme_counts.most_common():
            if count >= min_freq and phoneme not in self.phoneme_to_id:
                self.phoneme_to_id[phoneme] = idx
                self.id_to_phoneme[idx] = phoneme
                idx += 1
        
        self.vocab_built = True
        print(f"Built phonetic vocab: {len(self.phoneme_to_id)} tokens")
    
    def encode(self, text):
        """Text -> phoneme IDs."""
        phonemes = self.text_to_phonemes(text)
        return [self.phoneme_to_id.get(p, self.phoneme_to_id.get('<UNK>', 1)) for p in phonemes]
    
    def vocab_size(self):
        return len(self.phoneme_to_id)


def build_phonetic_bpe(corpus_path, vocab_size=2000):
    """
    Convert text → phonemes → apply BPE merging to reach target vocab size.
    This creates a phonetic tokenizer expanded to match BPE's vocab size.
    """
    # Step 1: Convert entire corpus to phoneme sequences
    phoneme_sequences = []
    phonetic = PhoneticTokenizer()
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            phonemes = phonetic.text_to_phonemes(line.strip())
            if phonemes:  # Only add non-empty sequences
                phoneme_sequences.append(' '.join(phonemes))
    
    # Step 2: Save phoneme sequences to temp file
    temp_file = 'temp_phonemes.txt'
    with open(temp_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(phoneme_sequences))
    
    # Step 3: Train BPE on phoneme sequences
    tokenizer = Tokenizer(BPE())
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        special_tokens=['<PAD>', '<UNK>', '<BOS>', '<EOS>']
    )
    tokenizer.train([temp_file], trainer)
    
    print(f"Built phonetic-BPE tokenizer: vocab_size={tokenizer.get_vocab_size()}")
    return tokenizer


if __name__ == '__main__':
    # Test basic phonetic
    phon = PhoneticTokenizer()
    phon.build_vocab('data/processed/train.txt')
    
    # Test phonetic-BPE expansion
    phon_bpe = build_phonetic_bpe('data/processed/train.txt', vocab_size=2000)
    print(f"Phonetic vocab: {phon.vocab_size()}")
    print(f"Phonetic-BPE vocab: {phon_bpe.get_vocab_size()}")