"""BPE tokenizer baseline using Hugging Face."""

from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.trainers import BpeTrainer


class BPETokenizer:
    def __init__(self, vocab_size=2000):
        self.tokenizer = Tokenizer(BPE())
        self.vocab_size_target = vocab_size
    
    def train(self, corpus_path):
        """Train BPE on corpus."""
        trainer = BpeTrainer(
            vocab_size=self.vocab_size_target,
            special_tokens=['<PAD>', '<UNK>', '<BOS>', '<EOS>']
        )
        self.tokenizer.train([corpus_path], trainer)
        print(f"Trained BPE tokenizer, vocab size: {self.tokenizer.get_vocab_size()}")
    
    def encode(self, text):
        """Text -> token IDs."""
        encoding = self.tokenizer.encode(text)
        return encoding.ids
    
    def vocab_size(self):
        return self.tokenizer.get_vocab_size()