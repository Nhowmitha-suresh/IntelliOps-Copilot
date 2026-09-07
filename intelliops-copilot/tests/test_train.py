import os
import torch
import pytest
from minigpt_config import ModelConfig, TrainConfig
from model import MiniGPT
from utils import save_checkpoint, load_checkpoint
from train import train, get_lr


def test_lr_schedule():
    lr = get_lr(step=0, warmup_iters=10, max_iters=100, lr=3e-4, min_lr=3e-5)
    assert lr > 0
    lr_max = get_lr(step=10, warmup_iters=10, max_iters=100, lr=3e-4, min_lr=3e-5)
    assert abs(lr_max - 3e-4) < 1e-6
    lr_end = get_lr(step=100, warmup_iters=10, max_iters=100, lr=3e-4, min_lr=3e-5)
    assert abs(lr_end - 3e-5) < 1e-6


def test_training_pipeline_tiny_synthetic(tmp_path):
    train_file = tmp_path / "train.txt"
    val_file = tmp_path / "val.txt"

    train_file.write_text("Hello world MiniGPT training pipeline test synthetic corpus text " * 20, encoding="utf-8")
    val_file.write_text("Hello world MiniGPT val corpus text " * 10, encoding="utf-8")

    m_cfg = ModelConfig(vocab_size=100, block_size=8, n_embd=16, n_head=2, n_layer=1)
    t_cfg = TrainConfig(batch_size=2, max_iters=10, eval_interval=5, eval_iters=2)

    model, tokenizer, train_losses, val_losses = train(
        train_txt_path=str(train_file),
        val_txt_path=str(val_file),
        tokenizer_path="non_existent_tokenizer.json",  # triggers fallback SimpleTokenizer
        checkpoint_dir=str(tmp_path),
        log_dir=str(tmp_path),
        plots_dir=str(tmp_path),
        model_cfg=m_cfg,
        train_cfg=t_cfg,
        verbose=False,
    )

    assert len(train_losses) > 0, "Train losses must not be empty"
    assert len(val_losses) > 0, "Val losses must not be empty"
    assert isinstance(train_losses[0], float)


def test_checkpoint_save_load_roundtrip(tmp_path):
    cfg = ModelConfig(vocab_size=50, block_size=8, n_embd=16, n_head=2, n_layer=1)
    model1 = MiniGPT(cfg)
    optimizer1 = torch.optim.AdamW(model1.parameters(), lr=1e-3)

    ckpt_file = str(tmp_path / "model_ckpt.pt")
    save_checkpoint(ckpt_file, model1, optimizer1, step=50, loss=2.5)
    assert os.path.exists(ckpt_file)

    # Instantiate fresh model and load weights
    model2 = MiniGPT(cfg)
    optimizer2 = torch.optim.AdamW(model2.parameters(), lr=1e-3)
    info = load_checkpoint(ckpt_file, model2, optimizer2)

    assert info["step"] == 50
    assert info["loss"] == 2.5

    # Check weight equality
    for p1, p2 in zip(model1.parameters(), model2.parameters()):
        assert torch.equal(p1, p2), "Model parameters must match after loading checkpoint"
