"""
Training loop, validation, metrics computation (loss, perplexity, top-1 & top-5 accuracy),
and checkpoint management.
"""

import time
import math
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Dict, List, Tuple

def compute_accuracy(logits: torch.Tensor, targets: torch.Tensor, topk: Tuple[int, ...] = (1, 5)) -> Dict[str, float]:
    """
    Computes top-k accuracy for next-word prediction.
    """
    maxk = max(topk)
    batch_size = targets.size(0)

    _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(targets.view(1, -1).expand_as(pred))

    res = {}
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        res[f"acc_top_{k}"] = (correct_k.item() / batch_size) * 100.0
    return res


def train_one_epoch(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    clip_grad: float = 5.0
) -> Dict[str, float]:
    """Runs a single training epoch."""
    model.train()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    num_batches = len(dataloader)

    for inputs, targets in dataloader:
        inputs = inputs.to(device)
        targets = targets.to(device)

        optimizer.zero_grad()
        logits, _ = model(inputs)
        loss = criterion(logits, targets)
        loss.backward()

        if clip_grad > 0:
            nn.utils.clip_grad_norm_(model.parameters(), clip_grad)

        optimizer.step()

        total_loss += loss.item()
        acc = compute_accuracy(logits, targets, topk=(1, 5))
        total_acc1 += acc["acc_top_1"]
        total_acc5 += acc["acc_top_5"]

    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20.0))  # numerical guard against overflow
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "acc_top_1": total_acc1 / num_batches,
        "acc_top_5": total_acc5 / num_batches,
    }


def evaluate(
    model: nn.Module,
    dataloader: torch.utils.data.DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Dict[str, float]:
    """Evaluates model on validation data."""
    model.eval()
    total_loss = 0.0
    total_acc1 = 0.0
    total_acc5 = 0.0
    num_batches = len(dataloader)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            logits, _ = model(inputs)
            loss = criterion(logits, targets)

            total_loss += loss.item()
            acc = compute_accuracy(logits, targets, topk=(1, 5))
            total_acc1 += acc["acc_top_1"]
            total_acc5 += acc["acc_top_5"]

    avg_loss = total_loss / num_batches
    perplexity = math.exp(min(avg_loss, 20.0))
    return {
        "loss": avg_loss,
        "perplexity": perplexity,
        "acc_top_1": total_acc1 / num_batches,
        "acc_top_5": total_acc5 / num_batches,
    }


def train_model(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    num_epochs: int = 10,
    lr: float = 0.002,
    device: torch.device = torch.device("cpu"),
    save_path: str = None
) -> Dict[str, List[float]]:
    """
    Trains a model across multiple epochs and records all metrics.
    """
    model.to(device)
    criterion = nn.CrossEntropyLoss(ignore_index=0)  # ignore padding token
    optimizer = optim.Adam(model.parameters(), lr=lr)

    history = {
        "train_loss": [],
        "train_perplexity": [],
        "train_acc_top_1": [],
        "train_acc_top_5": [],
        "val_loss": [],
        "val_perplexity": [],
        "val_acc_top_1": [],
        "val_acc_top_5": [],
        "epoch_times": [],
    }

    start_total = time.time()
    best_val_loss = float("inf")

    print(f"\n================ Training {model.name} ================")
    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()
        
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        epoch_duration = time.time() - epoch_start

        history["train_loss"].append(train_metrics["loss"])
        history["train_perplexity"].append(train_metrics["perplexity"])
        history["train_acc_top_1"].append(train_metrics["acc_top_1"])
        history["train_acc_top_5"].append(train_metrics["acc_top_5"])
        history["val_loss"].append(val_metrics["loss"])
        history["val_perplexity"].append(val_metrics["perplexity"])
        history["val_acc_top_1"].append(val_metrics["acc_top_1"])
        history["val_acc_top_5"].append(val_metrics["acc_top_5"])
        history["epoch_times"].append(epoch_duration)

        # Checkpointing
        if val_metrics["loss"] < best_val_loss and save_path:
            best_val_loss = val_metrics["loss"]
            torch.save(model.state_dict(), save_path)

        print(
            f"Epoch [{epoch:02d}/{num_epochs:02d}] ({epoch_duration:.2f}s) | "
            f"Train Loss: {train_metrics['loss']:.4f}, PPL: {train_metrics['perplexity']:.2f}, Top-1 Acc: {train_metrics['acc_top_1']:.2f}% | "
            f"Val Loss: {val_metrics['loss']:.4f}, PPL: {val_metrics['perplexity']:.2f}, Top-1 Acc: {val_metrics['acc_top_1']:.2f}%"
        )

    total_time = time.time() - start_total
    history["total_time"] = total_time
    print(f"Completed training {model.name} in {total_time:.2f}s (Best Val Loss: {best_val_loss:.4f})")
    return history
