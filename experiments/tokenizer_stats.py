"""Compare phonetic vs BPE tokenization stats."""

import sys
sys.path.insert(0, '.')

import csv
from pathlib import Path

from tokenizer.phonetic_g2p import PhoneticTokenizer, build_phonetic_bpe
from baselines.bpe_tokenizer import BPETokenizer

CORPUS = 'data/processed/train.txt'
OUT = Path('results/tables/tokenizer_stats.csv')


def count_tokens_in_corpus(tokenizer, corpus_path):
    """Total tokens, sentence count, and average tokens per sentence."""
    total_tokens = 0
    sentence_count = 0
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                total_tokens += len(tokenizer.encode(line))
                sentence_count += 1
    avg = total_tokens / sentence_count if sentence_count else 0.0
    return total_tokens, sentence_count, avg


def main():
    print("=== TOKENIZER STATS (50k subset) ===\n")

    print("Building phonetic base tokenizer (eSpeak symbols)...")
    phonetic_basic = PhoneticTokenizer()
    phonetic_basic.build_vocab(CORPUS)

    print("Building phonetic-BPE tokenizer (vocab_size=2000)...")
    phonetic_2000 = build_phonetic_bpe(CORPUS, vocab_size=2000)

    print("Building BPE tokenizer (vocab_size=2000)...")
    bpe_2000 = BPETokenizer(vocab_size=2000)
    bpe_2000.train(CORPUS)

    print("\nCounting tokens in corpus...")
    b_total, sent_count, b_avg = count_tokens_in_corpus(phonetic_basic, CORPUS)
    p_total, _, p_avg = count_tokens_in_corpus(phonetic_2000, CORPUS)
    bpe_total, _, bpe_avg = count_tokens_in_corpus(bpe_2000, CORPUS)

    print("\n" + "=" * 60)
    print(f"Corpus: {CORPUS}   Sentences: {sent_count}")
    print("=" * 60)
    print(f"{'Tokenizer':<25}{'Vocab Size':<15}{'Avg Tokens/Sent':<20}")
    print("-" * 60)
    print(f"{'Phonetic (basic)':<25}{phonetic_basic.vocab_size():<15}{b_avg:.2f}")
    print(f"{'Phonetic-BPE (2000)':<25}{phonetic_2000.get_vocab_size():<15}{p_avg:.2f}")
    print(f"{'BPE (2000)':<25}{bpe_2000.vocab_size():<15}{bpe_avg:.2f}")
    print("=" * 60)
    print(f"Phonetic-BPE / BPE: {p_avg / bpe_avg:.2f}x")
    print(f"Phonetic-basic / BPE: {b_avg / bpe_avg:.2f}x")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, 'w', newline='', encoding='utf-8') as f:
        w = csv.writer(f)
        w.writerow(['Tokenizer', 'Vocab Size', 'Avg Tokens/Sentence', 'Total Tokens', 'Sentences'])
        w.writerow(['Phonetic (basic)', phonetic_basic.vocab_size(), f'{b_avg:.2f}', b_total, sent_count])
        w.writerow(['Phonetic-BPE (2000)', phonetic_2000.get_vocab_size(), f'{p_avg:.2f}', p_total, sent_count])
        w.writerow(['BPE (2000)', bpe_2000.vocab_size(), f'{bpe_avg:.2f}', bpe_total, sent_count])
    print(f"\nSaved {OUT}")


if __name__ == '__main__':
    main()