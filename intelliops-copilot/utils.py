import os
import random
import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import List, Dict, Any


def set_seed(seed: int = 42) -> None:
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def save_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, step: int, loss: float) -> None:
    """Save model checkpoint to experiments/checkpoints/."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    checkpoint = {
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "loss": loss,
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(filepath: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer = None) -> Dict[str, Any]:
    """Load model checkpoint."""
    checkpoint = torch.load(filepath, map_location="cpu")
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer and checkpoint.get("optimizer_state_dict"):
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint


def plot_loss_curve(train_losses: List[float], eval_losses: List[float], save_path: str) -> None:
    """Plot training and evaluation loss curves and save to experiments/plots/."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label="Train Loss")
    if eval_losses:
        plt.plot(eval_losses, label="Eval Loss")
    plt.xlabel("Iterations")
    plt.ylabel("Loss")
    plt.title("MiniGPT Training Loss Curve")
    plt.legend()
    plt.grid(True)
    plt.savefig(save_path)
    plt.close()
