"""
Evaluation, metric aggregation, plotting, and qualitative analysis.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any
from models import count_parameters

def plot_training_curves(histories: Dict[str, Dict[str, List[float]]], save_path: str):
    """
    Plots comprehensive 4-panel comparison figure:
    1. Cross-Entropy Loss (Train & Val)
    2. Perplexity (Validation)
    3. Top-1 Prediction Accuracy (Validation)
    4. Training Time & Parameter Comparison
    """
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    epochs = range(1, len(next(iter(histories.values()))["val_loss"]) + 1)

    colors = {
        "Simple_RNN": "#e74c3c",       # Red
        "Bidirectional_RNN": "#e67e22", # Orange
        "LSTM": "#2ecc71",              # Green
        "GRU": "#3498db"                # Blue
    }

    # 1. Validation Loss
    ax1 = axes[0, 0]
    for name, hist in histories.items():
        c = colors.get(name, "black")
        display_name = name.replace("_", " ")
        ax1.plot(epochs, hist["val_loss"], marker='o', label=display_name, color=c, linewidth=2)
    ax1.set_title("Validation Loss across Epochs", fontsize=12, fontweight='bold')
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross-Entropy Loss")
    ax1.legend()
    ax1.grid(True, linestyle="--", alpha=0.6)

    # 2. Validation Perplexity
    ax2 = axes[0, 1]
    for name, hist in histories.items():
        c = colors.get(name, "black")
        display_name = name.replace("_", " ")
        ax2.plot(epochs, hist["val_perplexity"], marker='s', label=display_name, color=c, linewidth=2)
    ax2.set_title("Validation Perplexity (Lower is Better)", fontsize=12, fontweight='bold')
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Perplexity (exp(Loss))")
    ax2.legend()
    ax2.grid(True, linestyle="--", alpha=0.6)

    # 3. Validation Top-1 Accuracy
    ax3 = axes[1, 0]
    for name, hist in histories.items():
        c = colors.get(name, "black")
        display_name = name.replace("_", " ")
        ax3.plot(epochs, hist["val_acc_top_1"], marker='^', label=display_name, color=c, linewidth=2)
    ax3.set_title("Validation Next-Word Top-1 Accuracy (%)", fontsize=12, fontweight='bold')
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Accuracy (%)")
    ax3.legend()
    ax3.grid(True, linestyle="--", alpha=0.6)

    # 4. Total Training Time Bar Chart
    ax4 = axes[1, 1]
    names = [n.replace("_", " ") for n in histories.keys()]
    times = [h["total_time"] for h in histories.values()]
    bar_colors = [colors.get(n, "gray") for n in histories.keys()]
    bars = ax4.bar(names, times, color=bar_colors, width=0.55)
    ax4.set_title("Total Training Time (Seconds)", fontsize=12, fontweight='bold')
    ax4.set_ylabel("Seconds")
    for bar in bars:
        yval = bar.get_height()
        ax4.text(bar.get_x() + bar.get_width() / 2, yval + (max(times)*0.02), f"{yval:.1f}s", ha='center', va='bottom', fontsize=10, fontweight='bold')
    ax4.grid(True, axis='y', linestyle="--", alpha=0.6)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Training curves saved to: {save_path}")


def compute_diversity_metrics(text: str) -> Dict[str, float]:
    """
    Computes lexical diversity (Distinct-1, Distinct-2, Repetition rate).
    """
    words = text.lower().split()
    if not words:
        return {"distinct_1": 0.0, "distinct_2": 0.0, "repetition_ratio": 0.0}

    # Distinct-1
    distinct_1 = len(set(words)) / len(words)

    # Distinct-2
    bigrams = list(zip(words[:-1], words[1:])) if len(words) > 1 else []
    distinct_2 = len(set(bigrams)) / len(bigrams) if bigrams else 0.0

    # Repetition ratio (1 - unique / total)
    repetition_ratio = 1.0 - distinct_1

    return {
        "distinct_1": round(distinct_1, 4),
        "distinct_2": round(distinct_2, 4),
        "repetition_ratio": round(repetition_ratio, 4)
    }


def create_summary_dataframe(models: Dict[str, Any], histories: Dict[str, Dict[str, List[float]]]) -> pd.DataFrame:
    """
    Aggregates quantitative comparison table into a Pandas DataFrame.
    """
    summary_data = []

    for name, model in models.items():
        hist = histories[name]
        params = count_parameters(model)
        final_train_loss = hist["train_loss"][-1]
        final_val_loss = hist["val_loss"][-1]
        final_val_ppl = hist["val_perplexity"][-1]
        final_val_acc1 = hist["val_acc_top_1"][-1]
        final_val_acc5 = hist["val_acc_top_5"][-1]
        total_time = hist["total_time"]
        avg_epoch_time = sum(hist["epoch_times"]) / len(hist["epoch_times"])

        summary_data.append({
            "Model": name.replace("_", " "),
            "Parameters": f"{params:,}",
            "Train Loss": round(final_train_loss, 4),
            "Val Loss": round(final_val_loss, 4),
            "Val Perplexity": round(final_val_ppl, 2),
            "Val Top-1 Acc (%)": round(final_val_acc1, 2),
            "Val Top-5 Acc (%)": round(final_val_acc5, 2),
            "Total Time (s)": round(total_time, 2),
            "Sec/Epoch": round(avg_epoch_time, 2)
        })

    df = pd.DataFrame(summary_data)
    return df
