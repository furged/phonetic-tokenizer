"""Minimal GPT model for Hinglish language modeling."""

import torch
import torch.nn as nn
from torch.nn import functional as F

class GPTConfig:
    vocab_size = 2000
    max_seq_length = 256
    n_layers = 4
    n_heads = 4
    n_embed = 128
    dropout = 0.1
    learning_rate = 3e-4
    batch_size = 32
    num_epochs = 5
    device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Head(nn.Module):
    def __init__(self, head_size, config):
        super().__init__()
        self.key = nn.Linear(config.n_embed, head_size, bias=False)
        self.query = nn.Linear(config.n_embed, head_size, bias=False)
        self.value = nn.Linear(config.n_embed, head_size, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.register_buffer('tril', torch.tril(torch.ones(config.max_seq_length, config.max_seq_length)))
    
    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        v = self.value(x)
        
        scores = (q @ k.transpose(-2, -1)) * (1.0 / (k.shape[-1] ** 0.5))
        scores = scores.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        attn = F.softmax(scores, dim=-1)
        attn = self.dropout(attn)
        
        out = attn @ v
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_heads == 0
        head_size = config.n_embed // config.n_heads
        self.heads = nn.ModuleList([Head(head_size, config) for _ in range(config.n_heads)])
        self.proj = nn.Linear(config.n_embed, config.n_embed)
        self.dropout = nn.Dropout(config.dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.proj(out)
        out = self.dropout(out)
        return out


class FeedForward(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(config.n_embed, 4 * config.n_embed),
            nn.ReLU(),
            nn.Linear(4 * config.n_embed, config.n_embed),
            nn.Dropout(config.dropout),
        )
    
    def forward(self, x):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.sa = MultiHeadAttention(config)
        self.ffwd = FeedForward(config)
        self.ln1 = nn.LayerNorm(config.n_embed)
        self.ln2 = nn.LayerNorm(config.n_embed)
    
    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.token_embedding_table = nn.Embedding(config.vocab_size, config.n_embed)
        self.position_embedding_table = nn.Embedding(config.max_seq_length, config.n_embed)
        self.blocks = nn.Sequential(*[TransformerBlock(config) for _ in range(config.n_layers)])
        self.ln_f = nn.LayerNorm(config.n_embed)
        self.lm_head = nn.Linear(config.n_embed, config.vocab_size)
        
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=self.config.device))
        x = tok_emb + pos_emb
        
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, self.config.vocab_size), targets.view(-1))
        
        return logits, loss
    
    def generate(self, idx, max_tokens, temperature=1.0):
        for _ in range(max_tokens):
            idx_cond = idx[:, -self.config.max_seq_length:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :] / temperature
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx