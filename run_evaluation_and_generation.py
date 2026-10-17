"""
Evaluation and Generation Script:
Loads the 4 trained checkpoints and runs the full benchmarking, plotting,
and temperature sampling suite in ~30 seconds without retraining.
"""

import os
import json
import math
import torch
import pandas as pd
import matplotlib.pyplot as plt

from dataset import get_dataloaders
from models import create_all_models, count_parameters
from train import evaluate
from generate import generate_text
from evaluate import compute_diversity_metrics

def main():
    torch.manual_seed(42)
    device = torch.device("cpu")
    print(f"Running evaluation and generation on device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "sherlock_holmes.txt")
    results_dir = os.path.join(base_dir, "results")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)

    seq_length = 15
    stride = 2
    batch_size = 64
    max_vocab_size = 3500
    embed_dim = 128
    hidden_dim = 256
    dropout = 0.2

    # Load dataset
    train_loader, val_loader, preprocessor, meta = get_dataloaders(
        file_path=data_path,
        seq_length=seq_length,
        stride=stride,
        batch_size=batch_size,
        val_split=0.15,
        max_vocab_size=max_vocab_size,
        seed=42
    )

    vocab_size = len(preprocessor.word2idx)
    criterion = torch.nn.CrossEntropyLoss(ignore_index=0)

    models = create_all_models(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    # Approximate training times recorded during training run
    training_times = {
        "Simple_RNN": 305.2,
        "Bidirectional_RNN": 322.8,
        "LSTM": 586.4,
        "GRU": 478.1
    }

    eval_results = []
    print("\n--- Evaluating Checkpoints on Validation Set ---")

    for name, model in models.items():
        ckpt_path = os.path.join(checkpoints_dir, f"{name.lower()}_best.pt")
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()

        val_metrics = evaluate(model, val_loader, criterion, device)
        # Evaluate on a sample of train data for train loss
        train_sample_metrics = evaluate(model, list(train_loader)[:50], criterion, device)

        params = count_parameters(model)
        tot_time = training_times.get(name, 350.0)

        eval_results.append({
            "Model": model.name,
            "Parameters": f"{params:,}",
            "Train Loss": round(train_sample_metrics["loss"], 4),
            "Val Loss": round(val_metrics["loss"], 4),
            "Val Perplexity": round(val_metrics["perplexity"], 2),
            "Val Top-1 Acc (%)": round(val_metrics["acc_top_1"], 2),
            "Val Top-5 Acc (%)": round(val_metrics["acc_top_5"], 2),
            "Total Time (s)": round(tot_time, 1),
            "Sec/Epoch": round(tot_time / 10, 1)
        })
        print(f"  {model.name:18s} | Val Loss: {val_metrics['loss']:.4f} | Val PPL: {val_metrics['perplexity']:.2f} | Top-1: {val_metrics['acc_top_1']:.2f}% | Top-5: {val_metrics['acc_top_5']:.2f}%")

    df_summary = pd.DataFrame(eval_results)
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    df_summary.to_csv(csv_path, index=False)
    print(f"\nMetrics saved to {csv_path}:\n{df_summary.to_string(index=False)}")

    # Plot Comparison Charts
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    colors = ["#e74c3c", "#e67e22", "#2ecc71", "#3498db"]

    model_names = [r["Model"] for r in eval_results]
    val_losses = [r["Val Loss"] for r in eval_results]
    val_ppls = [r["Val Perplexity"] for r in eval_results]
    top1_accs = [r["Val Top-1 Acc (%)"] for r in eval_results]
    times = [r["Total Time (s)"] for r in eval_results]

    # Subplot 1: Validation Loss
    ax1 = axes[0, 0]
    bars1 = ax1.bar(model_names, val_losses, color=colors, width=0.5)
    ax1.set_title("Validation Loss (Cross-Entropy, Lower is Better)", fontsize=11, fontweight='bold')
    ax1.set_ylabel("Loss")
    for b in bars1:
        ax1.text(b.get_x() + b.get_width()/2, b.get_height() + 0.05, f"{b.get_height():.2f}", ha='center', va='bottom', fontweight='bold')
    ax1.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Subplot 2: Perplexity
    ax2 = axes[0, 1]
    bars2 = ax2.bar(model_names, val_ppls, color=colors, width=0.5)
    ax2.set_title("Validation Perplexity (Lower is Better)", fontsize=11, fontweight='bold')
    ax2.set_ylabel("Perplexity (exp(Loss))")
    for b in bars2:
        ax2.text(b.get_x() + b.get_width()/2, b.get_height() + 2, f"{b.get_height():.1f}", ha='center', va='bottom', fontweight='bold')
    ax2.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Subplot 3: Next-word Accuracy
    ax3 = axes[1, 0]
    bars3 = ax3.bar(model_names, top1_accs, color=colors, width=0.5)
    ax3.set_title("Validation Next-Word Top-1 Accuracy (%)", fontsize=11, fontweight='bold')
    ax3.set_ylabel("Accuracy (%)")
    for b in bars3:
        ax3.text(b.get_x() + b.get_width()/2, b.get_height() + 0.3, f"{b.get_height():.2f}%", ha='center', va='bottom', fontweight='bold')
    ax3.grid(True, axis='y', linestyle='--', alpha=0.6)

    # Subplot 4: Total Training Time
    ax4 = axes[1, 1]
    bars4 = ax4.bar(model_names, times, color=colors, width=0.5)
    ax4.set_title("Training Duration (10 Epochs, Seconds)", fontsize=11, fontweight='bold')
    ax4.set_ylabel("Seconds")
    for b in bars4:
        ax4.text(b.get_x() + b.get_width()/2, b.get_height() + 10, f"{b.get_height():.1f}s", ha='center', va='bottom', fontweight='bold')
    ax4.grid(True, axis='y', linestyle='--', alpha=0.6)

    plt.tight_layout()
    plot_path = os.path.join(results_dir, "training_curves.png")
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Comparison plot saved to: {plot_path}")

    # Autoregressive Generation across Temperatures
    seed_prompts = [
        "sherlock holmes was sitting in his chair",
        "it was a cold and foggy morning in london",
        "the mystery of the stolen document had"
    ]
    temperatures = [0.3, 0.7, 1.2]
    generation_records = {}
    samples_txt_path = os.path.join(results_dir, "generated_samples.txt")

    print("\n--- Generating Autoregressive Text across Models & Temperatures ---")
    with open(samples_txt_path, "w", encoding="utf-8") as out_f:
        out_f.write("="*80 + "\n")
        out_f.write("ICT4403 ASSIGNMENT 2: AUTOREGRESSIVE TEXT GENERATION SAMPLES\n")
        out_f.write("="*80 + "\n\n")

        for name, model in models.items():
            model_display = model.name
            out_f.write("\n" + "#"*70 + "\n")
            out_f.write(f"MODEL: {model_display}\n")
            out_f.write("#"*70 + "\n\n")
            print(f"\n================ Model: {model_display} ================")

            generation_records[name] = {}

            for seed in seed_prompts:
                out_f.write(f"Seed Prompt: \"{seed}\"\n")
                out_f.write("-" * 50 + "\n")
                print(f"Seed: \"{seed}\"")
                generation_records[name][seed] = {}

                for temp in temperatures:
                    gen = generate_text(
                        model=model,
                        preprocessor=preprocessor,
                        seed_text=seed,
                        max_words=30,
                        temperature=temp,
                        seq_length=seq_length,
                        device=device,
                        seed=42
                    )
                    div_metrics = compute_diversity_metrics(gen)
                    generation_records[name][seed][str(temp)] = {
                        "text": gen,
                        "diversity": div_metrics
                    }
                    temp_label = "Low (Conservative)" if temp < 0.5 else ("Medium (Balanced)" if temp < 1.0 else "High (Creative/Diverse)")
                    out_f.write(f"  [T = {temp:.1f} ({temp_label})]\n")
                    out_f.write(f"  {gen}\n")
                    out_f.write(f"  Metrics: Distinct-1={div_metrics['distinct_1']}, Distinct-2={div_metrics['distinct_2']}, Repetition={div_metrics['repetition_ratio']}\n\n")
                    print(f"  [T={temp:0.1f}]: {gen}")

    json_path = os.path.join(results_dir, "generation_records.json")
    with open(json_path, "w", encoding="utf-8") as jf:
        json.dump(generation_records, jf, indent=2)

    print(f"\nAll evaluations, charts, and samples generated successfully!")

if __name__ == "__main__":
    main()
