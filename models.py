"""
Neural network architectures for Autoregressive Next-Word Prediction:
1. Simple RNN (Elman RNN)
2. Bidirectional RNN
3. LSTM (Long Short-Term Memory)
4. GRU (Gated Recurrent Unit)
"""

import torch
import torch.nn as nn
from typing import Dict, Any

class SimpleRNNModel(nn.Module):
    """
    Standard Unidirectional Elman Recurrent Neural Network.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super(SimpleRNNModel, self).__init__()
        self.name = "Simple RNN"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.rnn = nn.RNN(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True,
            nonlinearity='tanh'
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        output, hidden = self.rnn(embedded, hidden)  # output: (batch_size, seq_len, hidden_dim)
        # Use final timestep output for next-word prediction
        last_step = output[:, -1, :]  # (batch_size, hidden_dim)
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)  # (batch_size, vocab_size)
        return logits, hidden


class BidirectionalRNNModel(nn.Module):
    """
    Bidirectional Recurrent Neural Network.
    Encodes the causal context window in both forward and backward directions.
    To ensure comparable parameter count, hidden_dim is split equally between directions.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super(BidirectionalRNNModel, self).__init__()
        self.name = "Bidirectional RNN"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        half_hidden = hidden_dim // 2
        self.rnn = nn.RNN(
            embed_dim,
            half_hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
            nonlinearity='tanh'
        )
        self.dropout = nn.Dropout(dropout)
        # Output features are half_hidden * 2 = hidden_dim
        self.fc = nn.Linear(half_hidden * 2, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)  # (batch_size, seq_len, embed_dim)
        output, hidden = self.rnn(embedded, hidden)  # (batch_size, seq_len, 2 * half_hidden)
        # Use final timestep representation of the sequence
        last_step = output[:, -1, :]  # (batch_size, hidden_dim)
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)
        return logits, hidden


class LSTMModel(nn.Module):
    """
    Long Short-Term Memory Network with input, forget, and output gates
    plus explicit cell state.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super(LSTMModel, self).__init__()
        self.name = "LSTM"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.lstm(embedded, hidden)  # output: (batch_size, seq_len, hidden_dim)
        last_step = output[:, -1, :]
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)
        return logits, hidden


class GRUModel(nn.Module):
    """
    Gated Recurrent Unit Network with update and reset gates.
    """
    def __init__(self, vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2):
        super(GRUModel, self).__init__()
        self.name = "GRU"
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(
            embed_dim,
            hidden_dim,
            num_layers=1,
            batch_first=True
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        last_step = output[:, -1, :]
        last_step = self.dropout(last_step)
        logits = self.fc(last_step)
        return logits, hidden


def count_parameters(model: nn.Module) -> int:
    """Returns total number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_all_models(vocab_size: int, embed_dim: int = 128, hidden_dim: int = 256, dropout: float = 0.2) -> Dict[str, nn.Module]:
    """
    Instantiates all 4 models under identical hyperparameter conditions.
    """
    models = {
        "Simple_RNN": SimpleRNNModel(vocab_size, embed_dim, hidden_dim, dropout),
        "Bidirectional_RNN": BidirectionalRNNModel(vocab_size, embed_dim, hidden_dim, dropout),
        "LSTM": LSTMModel(vocab_size, embed_dim, hidden_dim, dropout),
        "GRU": GRUModel(vocab_size, embed_dim, hidden_dim, dropout)
    }
    return models
