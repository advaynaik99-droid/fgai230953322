"""
Main execution pipeline for ICT4403 Assignment 2:
Autoregressive Text Generation System using Simple RNN, Bidirectional RNN, LSTM, and GRU.
"""

import os
import sys
import json
import torch
import pandas as pd
from dataset import get_dataloaders
from models import create_all_models, count_parameters
from train import train_model
from generate import generate_across_temperatures
from evaluate import plot_training_curves, create_summary_dataframe, compute_diversity_metrics

def main():
    # 1. Configuration & Reproducibility
    torch.manual_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing on hardware device: {device}")

    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "sherlock_holmes.txt")
    results_dir = os.path.join(base_dir, "results")
    checkpoints_dir = os.path.join(base_dir, "checkpoints")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(checkpoints_dir, exist_ok=True)

    # 2. Hyperparameters (Identical across all 4 models)
    seq_length = 15
    stride = 2
    batch_size = 64
    max_vocab_size = 3500
    embed_dim = 128
    hidden_dim = 256
    dropout = 0.2
    learning_rate = 0.002
    num_epochs = 10

    print("\n" + "="*70)
    print("TASK I: DATASET PREPROCESSING & SEQUENCE PREPARATION")
    print("="*70)
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
    print(f"Vocabulary size: {vocab_size}")

    # 3. Model Initialization
    models = create_all_models(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        hidden_dim=hidden_dim,
        dropout=dropout
    )

    print("\nModel Parameter Overview:")
    for name, model in models.items():
        print(f"  - {model.name}: {count_parameters(model):,} trainable parameters")

    # 4. Training All 4 Models Under Identical Experimental Conditions
    print("\n" + "="*70)
    print("TASK I (Cont.): MODEL TRAINING UNDER COMPARABLE CONDITIONS")
    print("="*70)
    histories = {}
    for name, model in models.items():
        save_path = os.path.join(checkpoints_dir, f"{name.lower()}_best.pt")
        hist = train_model(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            num_epochs=num_epochs,
            lr=learning_rate,
            device=device,
            save_path=save_path
        )
        histories[name] = hist

    # 5. Quantitative Evaluation & Plots (Task ii)
    print("\n" + "="*70)
    print("TASK II: QUANTITATIVE EVALUATION & BENCHMARKING")
    print("="*70)
    summary_df = create_summary_dataframe(models, histories)
    csv_path = os.path.join(results_dir, "metrics_summary.csv")
    summary_df.to_csv(csv_path, index=False)
    print(f"\nQuantitative Benchmark Summary (Saved to {csv_path}):")
    print(summary_df.to_string(index=False))

    plot_path = os.path.join(results_dir, "training_curves.png")
    plot_training_curves(histories, plot_path)

    # 6. Autoregressive Generation across Temperatures (Task iii)
    print("\n" + "="*70)
    print("TASK III: AUTOREGRESSIVE GENERATION & SAMPLING-TEMPERATURE ANALYSIS")
    print("="*70)
    seed_prompts = [
        "sherlock holmes was sitting in his chair",
        "it was a cold and foggy morning in london",
        "the mystery of the stolen document had"
    ]
    temperatures = [0.3, 0.7, 1.2]

    generation_records = {}
    samples_txt_path = os.path.join(results_dir, "generated_samples.txt")

    with open(samples_txt_path, "w", encoding="utf-8") as out_f:
        out_f.write("="*80 + "\n")
        out_f.write("ICT4403 ASSIGNMENT 2: AUTOREGRESSIVE TEXT GENERATION SAMPLES\n")
        out_f.write("="*80 + "\n\n")

        for name, model in models.items():
            model_display = model.name
            out_f.write("\n" + "#"*70 + "\n")
            out_f.write(f"MODEL: {model_display}\n")
            out_f.write("#"*70 + "\n\n")
            print(f"\n--- Model: {model_display} ---")

            generation_records[name] = {}

            for seed in seed_prompts:
                out_f.write(f"Seed Prompt: \"{seed}\"\n")
                out_f.write("-" * 50 + "\n")
                print(f"Seed: \"{seed}\"")
                generation_records[name][seed] = {}

                for temp in temperatures:
                    from generate import generate_text
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

    print(f"\nAll generated samples and temperature analysis logged to:")
    print(f"  - Text log: {samples_txt_path}")
    print(f"  - JSON log: {json_path}")
    print("\nPipeline execution completed successfully!")

if __name__ == "__main__":
    main()
