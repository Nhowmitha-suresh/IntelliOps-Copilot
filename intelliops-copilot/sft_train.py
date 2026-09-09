import os
import csv
import math
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Tuple, Dict, Any, Optional

from minigpt_config import ModelConfig, TrainConfig
from tokenizer import BPETokenizer, SimpleTokenizer
from dataset import TextDataset
from model import MiniGPT
from utils import set_seed, save_checkpoint, load_checkpoint, plot_loss_curve


def resolve_path(rel_path: str) -> str:
    if os.path.exists(rel_path):
        return rel_path
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path_from_base = os.path.join(base_dir, rel_path)
    if os.path.exists(path_from_base) or os.path.exists(os.path.dirname(path_from_base)):
        return path_from_base
    path_with_prefix = os.path.join("intelliops-copilot", rel_path)
    return path_with_prefix


def get_sft_lr(step: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    if step < warmup_iters:
        return lr * (step + 1) / max(1, warmup_iters)
    if step > max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_sft_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iters: int,
    device: str,
) -> Tuple[float, float, float, float]:
    model.eval()
    results = {}
    for split, loader in [("train", train_loader), ("val", val_loader)]:
        losses = []
        data_iter = iter(loader)
        for _ in range(eval_iters):
            try:
                x, y = next(data_iter)
            except StopIteration:
                data_iter = iter(loader)
                x, y = next(data_iter)
            x, y = x.to(device), y.to(device)
            _, loss = model(x, y)
            losses.append(loss.item())
        mean_loss = float(torch.tensor(losses).mean().item())
        ppl = math.exp(min(mean_loss, 20.0))
        results[split] = (mean_loss, ppl)

    model.train()
    return results["train"][0], results["val"][0], results["train"][1], results["val"][1]


def train_sft(
    sft_data_path: str = "data/processed/sft_data.txt",
    baseline_ckpt_path: str = "experiments/checkpoints/best_model.pt",
    tokenizer_path: str = "experiments/tokenizer.json",
    checkpoint_dir: str = "experiments/checkpoints_sft",
    log_dir: str = "experiments/logs",
    log_filename: str = "training_log_sft.csv",
    plots_dir: str = "experiments/plots_sft",
    learning_rate: float = 3e-5,
    max_iters: int = 1000,
    eval_interval: int = 100,
    eval_iters: int = 20,
    batch_size: int = 32,
    device: str = "cpu",
    verbose: bool = True,
):
    set_seed(42)

    sft_data_full = resolve_path(sft_data_path)
    baseline_ckpt_full = resolve_path(baseline_ckpt_path)
    if not os.path.exists(baseline_ckpt_full):
        baseline_ckpt_full = resolve_path("experiments/checkpoints/best.pt")
    tok_full = resolve_path(tokenizer_path)

    checkpoint_dir_full = resolve_path(checkpoint_dir)
    log_dir_full = resolve_path(log_dir)
    plots_dir_full = resolve_path(plots_dir)

    os.makedirs(checkpoint_dir_full, exist_ok=True)
    os.makedirs(log_dir_full, exist_ok=True)
    os.makedirs(plots_dir_full, exist_ok=True)

    # 1. Load Tokenizer
    if os.path.exists(tok_full):
        tokenizer = BPETokenizer.load(tok_full)
    else:
        tokenizer = SimpleTokenizer("MiniGPT fallback tokenizer text")

    # 2. Read SFT Data
    if not os.path.exists(sft_data_full):
        from sft_dataset import generate_sft_dataset
        sft_data_full = generate_sft_dataset()

    with open(sft_data_full, "r", encoding="utf-8") as f:
        sft_text = f.read()

    sft_ids = tokenizer.encode(sft_text)
    if verbose:
        print(f"SFT Dataset loaded: {len(sft_ids):,} tokens from {sft_data_full}")

    # Split 90% train, 10% val
    split_idx = int(0.9 * len(sft_ids))
    train_ids = sft_ids[:split_idx]
    val_ids = sft_ids[split_idx:]

    model_cfg = ModelConfig(vocab_size=max(tokenizer.vocab_size, 256))

    train_dataset = TextDataset(train_ids, block_size=model_cfg.block_size)
    val_dataset = TextDataset(val_ids, block_size=model_cfg.block_size)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # 3. Instantiate & Load Pretrained Model Baseline Checkpoint
    model = MiniGPT(model_cfg).to(device)

    if os.path.exists(baseline_ckpt_full):
        if verbose:
            print(f"Loading baseline starting checkpoint from: {baseline_ckpt_full}")
        load_checkpoint(baseline_ckpt_full, model)
    else:
        print(f"Warning: Baseline checkpoint not found at {baseline_ckpt_full}, fine-tuning from scratch!")

    # 4. Setup Optimizer with SFT Learning Rate (3e-5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    warmup_iters = 50
    min_lr = learning_rate / 10.0

    log_csv_path = os.path.join(log_dir_full, log_filename)
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "train_ppl", "val_ppl", "learning_rate", "elapsed_time_sec"])
    csv_file.flush()

    best_ckpt_path = os.path.join(checkpoint_dir_full, "best_model.pt")
    final_ckpt_path = os.path.join(checkpoint_dir_full, "checkpoint_final.pt")

    best_val_loss = float("inf")
    train_loss_history = []
    val_loss_history = []

    train_iter = iter(train_loader)
    start_time = time.time()

    if verbose:
        print(f"Starting SFT Fine-Tuning for {max_iters} steps at LR={learning_rate}...")

    model.train()
    pbar = tqdm(range(max_iters), desc="SFT Fine-Tuning", disable=not verbose)
    for step in pbar:
        lr = get_sft_lr(step, warmup_iters, max_iters, learning_rate, min_lr)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        try:
            x, y = next(train_iter)
        except StopIteration:
            train_iter = iter(train_loader)
            x, y = next(train_iter)

        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()
        logits, loss = model(x, y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == max_iters - 1:
            train_loss, val_loss, train_ppl, val_ppl = estimate_sft_loss(
                model, train_loader, val_loader, eval_iters, device
            )
            elapsed = time.time() - start_time

            csv_writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{train_ppl:.2f}", f"{val_ppl:.2f}", f"{lr:.6f}", f"{elapsed:.2f}"])
            csv_file.flush()

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)

            if verbose:
                pbar.set_postfix({
                    "tr_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "val_ppl": f"{val_ppl:.1f}",
                })

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_ckpt_path, model, optimizer, step, val_loss)
                save_checkpoint(os.path.join(checkpoint_dir_full, "best.pt"), model, optimizer, step, val_loss)

    csv_file.close()
    save_checkpoint(final_ckpt_path, model, optimizer, max_iters, train_loss_history[-1] if train_loss_history else 0.0)

    plot_path = os.path.join(plots_dir_full, "loss_curve_sft.png")
    plot_loss_curve(train_loss_history, val_loss_history, plot_path)

    if verbose:
        print(f"\nSFT Fine-Tuning Complete!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"SFT Log CSV saved to: {log_csv_path}")
        print(f"SFT Checkpoint saved to: {best_ckpt_path}")

    return model, tokenizer, train_loss_history, val_loss_history


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Fine-tune MiniGPT model on Shakespeare SFT dataset.")
    parser.add_argument("--max_iters", type=int, default=1000, help="Maximum fine-tuning steps")
    parser.add_argument("--lr", type=float, default=3e-5, help="Learning rate")
    args = parser.parse_args()

    train_sft(max_iters=args.max_iters, learning_rate=args.lr)
