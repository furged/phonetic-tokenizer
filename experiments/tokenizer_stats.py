"""Compare phonetic vs BPE tokenization stats."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from tokenizer.phonetic_g2p import PhoneticTokenizer, build_phonetic_bpe
from baselines.bpe_tokenizer import BPETokenizer
import csv

def count_tokens_in_corpus(tokenizer, corpus_path, is_bpe=False):
    """Count total and average tokens per sentence."""
    total_tokens = 0
    sentence_count = 0
    
    with open(corpus_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                if is_bpe:
                    tokens = tokenizer.encode(line)
                else:
                    tokens = tokenizer.encode(line)
                total_tokens += len(tokens)
                sentence_count += 1
    
    avg_tokens = total_tokens / sentence_count if sentence_count > 0 else 0
    return total_tokens, sentence_count, avg_tokens


def compare_tokenizers(corpus_path, output_csv='results/tables/tokenizer_stats.csv'):
    """Compare phonetic and BPE tokenizers."""
    print("=== TOKENIZER COMPARISON ===\n")
    
    # Build phonetic tokenizer (basic, ~40 tokens)
    print("Building basic phonetic tokenizer...")
    phonetic_basic = PhoneticTokenizer()
    phonetic_basic.build_vocab(corpus_path)
    
    # Build phonetic-BPE (expanded to 2000)
    print("Building phonetic-BPE tokenizer (vocab_size=2000)...")
    phonetic_2000 = build_phonetic_bpe(corpus_path, vocab_size=2000)
    
    # Build BPE (2000 tokens)
    print("Building BPE tokenizer (vocab_size=2000)...")
    bpe_2000 = BPETokenizer(vocab_size=2000)
    bpe_2000.train(corpus_path)
    
    # Count tokens
    print("\nCounting tokens in corpus...")
    phon_basic_total, sent_count, phon_basic_avg = count_tokens_in_corpus(phonetic_basic, corpus_path)
    phon_2000_total, _, phon_2000_avg = count_tokens_in_corpus(phonetic_2000, corpus_path, is_bpe=True)
    bpe_2000_total, _, bpe_2000_avg = count_tokens_in_corpus(bpe_2000, corpus_path, is_bpe=True)
    
    # Print results
    print("\n" + "="*60)
    print("TOKENIZER STATS COMPARISON")
    print("="*60)
    print(f"Corpus: {corpus_path}")
    print(f"Total sentences: {sent_count}\n")
    
    print(f"{'Tokenizer':<25} {'Vocab Size':<15} {'Avg Tokens/Sent':<20}")
    print("-" * 60)
    print(f"{'Phonetic (basic)':<25} {phonetic_basic.vocab_size():<15} {phon_basic_avg:.2f}")
    print(f"{'Phonetic-BPE (2000)':<25} {phonetic_2000.get_vocab_size():<15} {phon_2000_avg:.2f}")
    print(f"{'BPE (2000)':<25} {bpe_2000.vocab_size():<15} {bpe_2000_avg:.2f}")
    print("="*60)
    
    # Compactness ratio (lower is more compact)
    print(f"\nCompactness (tokens/sentence):")
    print(f"  Phonetic-BPE / BPE: {phon_2000_avg / bpe_2000_avg:.2f}x")
    print(f"  Phonetic-Basic / BPE: {phon_basic_avg / bpe_2000_avg:.2f}x")
    
    # Save to CSV
    results = [
        ['Tokenizer', 'Vocab Size', 'Avg Tokens/Sentence', 'Total Tokens', 'Sentences'],
        ['Phonetic (basic)', phonetic_basic.vocab_size(), f'{phon_basic_avg:.2f}', phon_basic_total, sent_count],
        ['Phonetic-BPE (2000)', phonetic_2000.get_vocab_size(), f'{phon_2000_avg:.2f}', phon_2000_total, sent_count],
        ['BPE (2000)', bpe_2000.vocab_size(), f'{bpe_2000_avg:.2f}', bpe_2000_total, sent_count],
    ]
    
    with open(output_csv, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerows(results)
    
    print(f"\nResults saved to {output_csv}")


if __name__ == '__main__':
    compare_tokenizers('data/processed/train.txt')