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


def get_lr(step: int, warmup_iters: int, max_iters: int, lr: float, min_lr: float) -> float:
    """
    Computes learning rate with linear warmup and cosine decay.
    """
    if step < warmup_iters:
        return lr * (step + 1) / max(1, warmup_iters)
    if step > max_iters:
        return min_lr
    decay_ratio = (step - warmup_iters) / max(1, max_iters - warmup_iters)
    assert 0.0 <= decay_ratio <= 1.0
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (lr - min_lr)


@torch.no_grad()
def estimate_loss(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    eval_iters: int,
    device: str,
) -> Tuple[float, float, float, float]:
    """
    Evaluates average train and validation loss/perplexity over eval_iters batches.
    """
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
        ppl = math.exp(min(mean_loss, 20.0))  # Cap for numerical safety
        results[split] = (mean_loss, ppl)

    model.train()
    return results["train"][0], results["val"][0], results["train"][1], results["val"][1]


def train(
    train_txt_path: str = "data/processed/train.txt",
    val_txt_path: str = "data/processed/val.txt",
    tokenizer_path: str = "experiments/tokenizer.json",
    checkpoint_dir: str = "experiments/checkpoints",
    log_dir: str = "experiments/logs",
    plots_dir: str = "experiments/plots",
    model_cfg: Optional[ModelConfig] = None,
    train_cfg: Optional[TrainConfig] = None,
    verbose: bool = True,
):
    """
    End-to-end MiniGPT training pipeline.
    """
    set_seed(42)
    if train_cfg is None:
        train_cfg = TrainConfig()

    device = train_cfg.device

    # 1. Load Tokenizer
    if os.path.exists(tokenizer_path):
        if verbose:
            print(f"Loading BPETokenizer from {tokenizer_path}...")
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        if verbose:
            print("Tokenizer not found, creating fallback SimpleTokenizer...")
        tokenizer = SimpleTokenizer("MiniGPT fallback dataset text")

    # Read train & val corpora
    if os.path.exists(train_txt_path):
        with open(train_txt_path, "r", encoding="utf-8") as f:
            train_text = f.read()
    else:
        train_text = "MiniGPT train text sample " * 1000

    if os.path.exists(val_txt_path):
        with open(val_txt_path, "r", encoding="utf-8") as f:
            val_text = f.read()
    else:
        val_text = "MiniGPT val text sample " * 200

    train_ids = tokenizer.encode(train_text)
    val_ids = tokenizer.encode(val_text)

    if verbose:
        print(f"Train tokens: {len(train_ids):,}, Val tokens: {len(val_ids):,}")

    # 2. Setup Config & Datasets
    if model_cfg is None:
        model_cfg = ModelConfig(vocab_size=max(tokenizer.vocab_size, 256))
    else:
        model_cfg.vocab_size = max(tokenizer.vocab_size, 256)

    train_dataset = TextDataset(train_ids, block_size=model_cfg.block_size)
    val_dataset = TextDataset(val_ids, block_size=model_cfg.block_size)

    if len(train_dataset) == 0 or len(val_dataset) == 0:
        raise ValueError("Corpus too small for specified block_size")

    train_loader = DataLoader(train_dataset, batch_size=train_cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=train_cfg.batch_size, shuffle=False)

    # Instantiate Model
    model = MiniGPT(model_cfg).to(device)
    if verbose:
        n_params = model.get_num_params()
        print(f"Model instantiated on {device}: {n_params:,} parameters ({n_params / 1e6:.2f}M)")

    # 3. Optimizer & LR Schedule Setup
    optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate, weight_decay=0.01)
    warmup_iters = min(100, max(1, train_cfg.max_iters // 10))
    min_lr = train_cfg.learning_rate / 10.0

    # Ensure log directories exist
    log_csv_path = os.path.join(log_dir, "training_log.csv")
    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)

    # Initialize CSV log file
    csv_file = open(log_csv_path, "w", newline="", encoding="utf-8")
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["step", "train_loss", "val_loss", "train_ppl", "val_ppl", "learning_rate", "elapsed_time_sec"])
    csv_file.flush()

    train_loss_history = []
    val_loss_history = []
    eval_steps_history = []
    best_val_loss = float("inf")
    best_ckpt_path = os.path.join(checkpoint_dir, "best_model.pt")
    final_ckpt_path = os.path.join(checkpoint_dir, "checkpoint_final.pt")

    train_iter = iter(train_loader)
    start_time = time.time()

    if verbose:
        print(f"Starting training for {train_cfg.max_iters} iterations...")

    model.train()
    pbar = tqdm(range(train_cfg.max_iters), desc="Training MiniGPT", disable=not verbose)
    for step in pbar:
        # LR Schedule step
        lr = get_lr(step, warmup_iters, train_cfg.max_iters, train_cfg.learning_rate, min_lr)
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

        # Evaluation & Logging
        if step % train_cfg.eval_interval == 0 or step == train_cfg.max_iters - 1:
            train_loss, val_loss, train_ppl, val_ppl = estimate_loss(
                model, train_loader, val_loader, train_cfg.eval_iters, device
            )
            elapsed = time.time() - start_time

            csv_writer.writerow([step, f"{train_loss:.4f}", f"{val_loss:.4f}", f"{train_ppl:.2f}", f"{val_ppl:.2f}", f"{lr:.6f}", f"{elapsed:.2f}"])
            csv_file.flush()

            train_loss_history.append(train_loss)
            val_loss_history.append(val_loss)
            eval_steps_history.append(step)

            if verbose:
                pbar.set_postfix({
                    "tr_loss": f"{train_loss:.4f}",
                    "val_loss": f"{val_loss:.4f}",
                    "tr_ppl": f"{train_ppl:.1f}",
                    "val_ppl": f"{val_ppl:.1f}",
                })

            # Save best validation loss checkpoint
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                save_checkpoint(best_ckpt_path, model, optimizer, step, val_loss)

    csv_file.close()

    # Save final checkpoint
    save_checkpoint(final_ckpt_path, model, optimizer, train_cfg.max_iters, train_loss_history[-1] if train_loss_history else 0.0)

    # Save loss plot
    plot_path = os.path.join(plots_dir, "loss_curve.png")
    plot_loss_curve(train_loss_history, val_loss_history, plot_path)

    if verbose:
        print(f"\nTraining Complete!")
        print(f"Best Val Loss: {best_val_loss:.4f}")
        print(f"Log CSV saved to: {log_csv_path}")
        print(f"Best checkpoint saved to: {best_ckpt_path}")
        print(f"Final checkpoint saved to: {final_ckpt_path}")
        print(f"Loss plot saved to: {plot_path}")

    return model, tokenizer, train_loss_history, val_loss_history


if __name__ == "__main__":
    # Short run with max_iters=500 to verify pipeline end-to-end
    cfg_train = TrainConfig(max_iters=500, eval_interval=100, eval_iters=10)
    train(train_cfg=cfg_train)
