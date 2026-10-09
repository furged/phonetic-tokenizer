"""Training script for nanoGPT with phonetic or BPE tokenizers."""

import sys
sys.path.insert(0, '.')

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import numpy as np
from pathlib import Path

from model.nanogpt import GPT, GPTConfig
from tokenizer.phonetic_g2p import PhoneticTokenizer, build_phonetic_bpe
from baselines.bpe_tokenizer import BPETokenizer


class HinglishDataset(Dataset):
    def __init__(self, corpus_path, tokenizer, max_seq_length=256, max_samples=None):
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        
        # Read corpus
        with open(corpus_path, 'r', encoding='utf-8') as f:
            sentences = [line.strip() for line in f if line.strip()]
        
        if max_samples:
            sentences = sentences[:max_samples]
        
        # Tokenize all sentences
        self.token_ids = []
        for sent in sentences:
            if hasattr(tokenizer, 'encode'):
                # BPE tokenizer
                ids = tokenizer.encode(sent)
            else:
                # Phonetic tokenizer (huggingface)
                encoding = tokenizer.encode(sent)
                ids = encoding.ids
            
            self.token_ids.extend(ids)
        
        # Create sequences
        self.sequences = []
        for i in range(len(self.token_ids) - max_seq_length):
            seq = self.token_ids[i:i + max_seq_length]
            target = self.token_ids[i + 1:i + max_seq_length + 1]
            self.sequences.append((seq, target))
    
    def __len__(self):
        return len(self.sequences)
    
    def __getitem__(self, idx):
        seq, target = self.sequences[idx]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)


def train_model(tokenizer_name, corpus_path='data/processed/train.txt'):
    """Train GPT with given tokenizer."""
    
    print(f"\n{'='*60}")
    print(f"Training with {tokenizer_name.upper()} tokenizer")
    print(f"{'='*60}\n")
    
    # Setup
    config = GPTConfig()
    device = config.device
    print(f"Device: {device}\n")
    
    # Build tokenizer
    if tokenizer_name == 'phonetic':
        print("Building phonetic tokenizer...")
        tokenizer = build_phonetic_bpe(corpus_path, vocab_size=2000)
        config.vocab_size = tokenizer.get_vocab_size()
    else:  # bpe
        print("Building BPE tokenizer...")
        tokenizer = BPETokenizer(vocab_size=2000)
        tokenizer.train(corpus_path)
        config.vocab_size = tokenizer.vocab_size()
    
    print(f"Vocab size: {config.vocab_size}\n")
    
    # Create dataset and dataloader
    print("Creating dataset...")
    dataset = HinglishDataset(corpus_path, tokenizer, max_seq_length=config.max_seq_length)
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True)
    print(f"Dataset size: {len(dataset)} sequences\n")
    
    # Build model
    print("Building GPT model...")
    model = GPT(config).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}\n")
    
    # Training loop
    print("Starting training...\n")
    best_loss = float('inf')
    
    for epoch in range(config.num_epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_idx, (x, y) in enumerate(dataloader):
            x, y = x.to(device), y.to(device)
            
            logits, loss = model(x, y)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
            if (batch_idx + 1) % 10 == 0:
                avg_loss = epoch_loss / (batch_idx + 1)
                print(f"Epoch {epoch+1}/{config.num_epochs}, Batch {batch_idx+1}/{len(dataloader)}, Loss: {avg_loss:.4f}")
        
        avg_epoch_loss = epoch_loss / len(dataloader)
        perplexity = torch.exp(torch.tensor(avg_epoch_loss)).item()
        
        print(f"\nEpoch {epoch+1} - Loss: {avg_epoch_loss:.4f}, Perplexity: {perplexity:.4f}\n")
        
        if avg_epoch_loss < best_loss:
            best_loss = avg_epoch_loss
            torch.save(model.state_dict(), f'results/model_{tokenizer_name}_best.pt')
            print(f"Saved best model (loss: {best_loss:.4f})\n")
    
    # Save final model
    torch.save(model.state_dict(), f'results/model_{tokenizer_name}_final.pt')
    print(f"Training complete. Final loss: {avg_epoch_loss:.4f}")
    
    return avg_epoch_loss, perplexity


if __name__ == '__main__':
    # Train both
    results = {}
    
    for tokenizer_type in ['bpe', 'phonetic']:
        loss, perp = train_model(tokenizer_type)
        results[tokenizer_type] = {'loss': loss, 'perplexity': perp}
    
    # Summary
    print(f"\n{'='*60}")
    print("TRAINING SUMMARY")
    print(f"{'='*60}")
    for tok, metrics in results.items():
        print(f"{tok.upper()}: Loss={metrics['loss']:.4f}, Perplexity={metrics['perplexity']:.4f}")