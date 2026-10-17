"""
Autoregressive text generation with temperature scaling and seed conditioning.
"""

import re
import torch
import torch.nn.functional as F
from typing import List, Dict, Any
from dataset import TextPreprocessor

def detokenize(tokens: List[str]) -> str:
    """
    Reconstructs human-readable English text from token list,
    handling punctuation spacing and sentence capitalization.
    """
    text = " ".join(tokens)
    # Remove space before punctuation
    text = re.sub(r'\s+([.,!?;:\'\)])', r'\1', text)
    # Remove space after opening parenthesis or quotes
    text = re.sub(r'(\()\s+', r'\1', text)
    # Fix contractions (e.g., don ' t -> don't, it ' s -> it's)
    text = re.sub(r"\s*'\s*", "'", text)
    # Capitalize the first letter and letters following periods
    sentences = re.split(r'([.!?]\s+)', text)
    capitalized = "".join([s.capitalize() if i % 2 == 0 else s for i, s in enumerate(sentences)])
    return capitalized.strip()


def generate_text(
    model: torch.nn.Module,
    preprocessor: TextPreprocessor,
    seed_text: str,
    max_words: int = 30,
    temperature: float = 1.0,
    seq_length: int = 15,
    device: torch.device = torch.device("cpu"),
    seed: int = None
) -> str:
    """
    Generates text autoregressively word-by-word given a seed prompt.

    Args:
        model: Trained PyTorch RNN/LSTM/GRU model.
        preprocessor: TextPreprocessor instance with word mappings.
        seed_text: Seed string prompt.
        max_words: Number of new words to generate sequentially.
        temperature: Sampling temperature (e.g., 0.3, 0.7, 1.2).
        seq_length: Expected context window length.
        device: CPU or CUDA device.
        seed: Optional random seed for reproducible sampling.
    """
    if seed is not None:
        torch.manual_seed(seed)

    model.eval()
    seed_tokens = preprocessor.clean_and_tokenize(seed_text)
    if not seed_tokens:
        seed_tokens = ["the"]

    generated_tokens = list(seed_tokens)

    with torch.no_grad():
        for _ in range(max_words):
            # Take the trailing context window up to seq_length
            current_window = generated_tokens[-seq_length:]
            indices = preprocessor.tokens_to_indices(current_window)

            # Pad if shorter than seq_length
            if len(indices) < seq_length:
                indices = [preprocessor.pad_idx] * (seq_length - len(indices)) + indices

            input_tensor = torch.tensor([indices], dtype=torch.long, device=device)
            logits, _ = model(input_tensor)  # (1, vocab_size)

            # Mask out special tokens <pad> and <unk> from being generated
            logits[0, preprocessor.pad_idx] = -float("inf")
            logits[0, preprocessor.unk_idx] = -float("inf")

            if temperature <= 0.05:
                # Greedy selection
                next_idx = torch.argmax(logits, dim=-1).item()
            else:
                # Temperature scaling
                scaled_logits = logits / temperature
                probs = F.softmax(scaled_logits, dim=-1)
                next_idx = torch.multinomial(probs, num_samples=1).item()

            next_word = preprocessor.idx2word.get(next_idx, preprocessor.unk_token)
            generated_tokens.append(next_word)

    return detokenize(generated_tokens)


def generate_across_temperatures(
    model: torch.nn.Module,
    preprocessor: TextPreprocessor,
    seed_prompts: List[str],
    temperatures: List[float] = [0.3, 0.7, 1.2],
    max_words: int = 30,
    seq_length: int = 15,
    device: torch.device = torch.device("cpu")
) -> Dict[str, Dict[float, str]]:
    """
    Generates text across a matrix of seed prompts and temperature values.
    """
    results = {}
    for seed in seed_prompts:
        results[seed] = {}
        for temp in temperatures:
            gen_text = generate_text(
                model=model,
                preprocessor=preprocessor,
                seed_text=seed,
                max_words=max_words,
                temperature=temp,
                seq_length=seq_length,
                device=device,
                seed=42  # consistent seed per temperature for clean comparison
            )
            results[seed][temp] = gen_text
    return results
