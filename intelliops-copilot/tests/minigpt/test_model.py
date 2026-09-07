import math
import torch
import pytest
from minigpt.minigpt_config import ModelConfig
from minigpt.model import MiniGPT, CausalSelfAttention, FeedForward, TransformerBlock


def test_forward_pass_output_shape():
    batch_size = 4
    block_size = 16
    vocab_size = 100

    cfg = ModelConfig(vocab_size=vocab_size, block_size=block_size, n_embd=32, n_head=2, n_layer=2)
    model = MiniGPT(cfg)

    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    logits, loss = model(idx)

    assert logits.shape == (batch_size, block_size, vocab_size), f"Expected shape {(batch_size, block_size, vocab_size)}, got {logits.shape}"
    assert loss is None, "Loss should be None when targets are not provided"


def test_causal_mask_blocks_future_positions():
    block_size = 8
    n_embd = 32
    n_head = 4
    cfg = ModelConfig(vocab_size=50, block_size=block_size, n_embd=n_embd, n_head=n_head)

    attn = CausalSelfAttention(cfg)
    attn.eval()  # Disable dropout for deterministic checking

    x = torch.randn(1, block_size, n_embd)
    _, att_weights = attn(x, return_attn_weights=True)  # Shape: (1, n_head, T, T)

    assert att_weights.shape == (1, n_head, block_size, block_size)

    # Verify that for any position i, attention to position j > i is strictly 0.0
    for head_idx in range(n_head):
        for i in range(block_size):
            for j in range(i + 1, block_size):
                attn_val = att_weights[0, head_idx, i, j].item()
                assert attn_val == 0.0, f"Causal mask failed: Head {head_idx}, pos {i} attended to future pos {j} with weight {attn_val}"


def test_loss_computation_with_targets():
    batch_size = 2
    block_size = 12
    vocab_size = 64

    cfg = ModelConfig(vocab_size=vocab_size, block_size=block_size, n_embd=32, n_head=2, n_layer=2)
    model = MiniGPT(cfg)

    idx = torch.randint(0, vocab_size, (batch_size, block_size))
    targets = torch.randint(0, vocab_size, (batch_size, block_size))

    logits, loss = model(idx, targets=targets)

    assert logits.shape == (batch_size, block_size, vocab_size)
    assert loss is not None, "Loss must not be None when targets are provided"
    assert loss.ndim == 0, "Loss must be a 0D scalar tensor"
    assert loss.item() > 0, "Loss must be positive"

    # Expected initial loss for random weights is around -ln(1/vocab_size)
    expected_loss_approx = math.log(vocab_size)
    assert abs(loss.item() - expected_loss_approx) < 2.0, f"Initial loss {loss.item()} should be near expected log(vocab_size) {expected_loss_approx}"
