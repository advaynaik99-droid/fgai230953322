"""
Dataset loading, cleaning, tokenization, vocabulary creation,
and sequence generation for Autoregressive Next-Word Prediction.
"""

import os
import re
import string
from collections import Counter
from typing import Tuple, List, Dict
import torch
from torch.utils.data import Dataset, DataLoader

class TextPreprocessor:
    """
    Cleans raw text, extracts vocabulary, and maps tokens to integers.
    """
    def __init__(self, max_vocab_size: int = 3500, min_freq: int = 2):
        self.max_vocab_size = max_vocab_size
        self.min_freq = min_freq
        self.word2idx: Dict[str, int] = {}
        self.idx2word: Dict[int, str] = {}
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.pad_idx = 0
        self.unk_idx = 1

    @staticmethod
    def strip_gutenberg_headers(text: str) -> str:
        """Removes Project Gutenberg preamble and license disclaimers."""
        start_pattern = r"\*\*\* START OF THE PROJECT GUTENBERG EBOOK .+ \*\*\*"
        end_pattern = r"\*\*\* END OF THE PROJECT GUTENBERG EBOOK .+ \*\*\*"
        
        start_match = re.search(start_pattern, text)
        if start_match:
            text = text[start_match.end():]
            
        end_match = re.search(end_pattern, text)
        if end_match:
            text = text[:end_match.start()]
            
        return text.strip()

    @staticmethod
    def clean_and_tokenize(text: str) -> List[str]:
        """
        Tokenizes text into words and punctuation while preserving semantic structure.
        """
        # Convert to lowercase
        text = text.lower()
        # Separate punctuation with spaces so they become distinct tokens or clean
        # Replace non-standard whitespace and dashes
        text = re.sub(r'[\r\n\t]+', ' ', text)
        text = re.sub(r'—|–', ' ', text)
        # Separate punctuation marks
        text = re.sub(r"([.,!?;:\(\)\"\'])", r" \1 ", text)
        # Collapse multiple spaces
        text = re.sub(r'\s+', ' ', text).strip()
        tokens = text.split()
        return tokens

    def build_vocab(self, tokens: List[str]):
        """Builds vocabulary mapping with frequency thresholds."""
        counter = Counter(tokens)
        
        # Reserved special tokens
        self.word2idx = {self.pad_token: self.pad_idx, self.unk_token: self.unk_idx}
        self.idx2word = {self.pad_idx: self.pad_token, self.unk_idx: self.unk_token}
        
        # Filter by min_freq and limit to max_vocab_size
        vocab_words = [word for word, count in counter.most_common() if count >= self.min_freq]
        vocab_words = vocab_words[:self.max_vocab_size - len(self.word2idx)]
        
        for idx, word in enumerate(vocab_words, start=len(self.word2idx)):
            self.word2idx[word] = idx
            self.idx2word[idx] = word
            
        print(f"Vocabulary constructed: {len(self.word2idx)} unique tokens (including <pad>, <unk>).")
        return self.word2idx, self.idx2word

    def tokens_to_indices(self, tokens: List[str]) -> List[int]:
        """Converts tokens into index integers."""
        return [self.word2idx.get(token, self.unk_idx) for token in tokens]

    def indices_to_tokens(self, indices: List[int]) -> List[str]:
        """Converts indices back into string tokens."""
        return [self.idx2word.get(idx, self.unk_token) for idx in indices]


class AutoregressiveTextDataset(Dataset):
    """
    Sliding window dataset for next-word prediction:
    Given sequence [w_1, w_2, ..., w_L], target is w_{L+1}.
    """
    def __init__(self, token_indices: List[int], seq_length: int = 15, stride: int = 2):
        self.seq_length = seq_length
        self.inputs = []
        self.targets = []
        
        for i in range(0, len(token_indices) - seq_length, stride):
            self.inputs.append(token_indices[i : i + seq_length])
            self.targets.append(token_indices[i + seq_length])
            
        self.inputs = torch.tensor(self.inputs, dtype=torch.long)
        self.targets = torch.tensor(self.targets, dtype=torch.long)
        
    def __len__(self):
        return len(self.inputs)
        
    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def get_dataloaders(
    file_path: str,
    seq_length: int = 15,
    stride: int = 2,
    batch_size: int = 64,
    val_split: float = 0.15,
    max_vocab_size: int = 3500,
    seed: int = 42
) -> Tuple[DataLoader, DataLoader, TextPreprocessor, Dict[str, any]]:
    """
    Loads raw text, runs preprocessing, splits into train/val datasets,
    and returns DataLoaders along with dataset metadata.
    """
    with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
        raw_text = f.read()
        
    preprocessor = TextPreprocessor(max_vocab_size=max_vocab_size)
    cleaned_text = preprocessor.strip_gutenberg_headers(raw_text)
    tokens = preprocessor.clean_and_tokenize(cleaned_text)
    
    # Dataset statistics
    total_tokens = len(tokens)
    unique_tokens = len(set(tokens))
    lexical_diversity = unique_tokens / total_tokens if total_tokens > 0 else 0
    
    preprocessor.build_vocab(tokens)
    token_indices = preprocessor.tokens_to_indices(tokens)
    
    dataset = AutoregressiveTextDataset(token_indices, seq_length=seq_length, stride=stride)
    total_samples = len(dataset)
    
    val_size = int(total_samples * val_split)
    train_size = total_samples - val_size
    
    generator = torch.Generator().manual_seed(seed)
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size], generator=generator
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, drop_last=False)
    
    metadata = {
        "raw_characters": len(raw_text),
        "cleaned_characters": len(cleaned_text),
        "total_tokens": total_tokens,
        "unique_tokens": unique_tokens,
        "lexical_diversity": lexical_diversity,
        "vocab_size": len(preprocessor.word2idx),
        "seq_length": seq_length,
        "stride": stride,
        "total_sequences": total_samples,
        "train_samples": train_size,
        "val_samples": val_size
    }
    
    print(f"Dataset stats: {total_tokens} tokens, {total_samples} sequences created.")
    print(f"Train batches: {len(train_loader)}, Validation batches: {len(val_loader)}")
    
    return train_loader, val_loader, preprocessor, metadata
