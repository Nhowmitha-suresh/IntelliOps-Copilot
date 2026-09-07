import torch
import pytest
from minigpt_config import ModelConfig, TrainConfig
from tokenizer import SimpleTokenizer
from model import MiniGPT
from dataset import TextDataset


def test_model_config_defaults():
    cfg = ModelConfig()
    assert cfg.n_embd == 128
    assert cfg.n_head == 4
    assert cfg.n_layer == 4
    assert cfg.block_size == 128
    assert cfg.dropout == 0.1
    assert cfg.vocab_size == 50257


def test_train_config_defaults():
    cfg = TrainConfig()
    assert cfg.batch_size == 32
    assert cfg.learning_rate == 3e-4
    assert cfg.max_iters == 5000
    assert cfg.eval_interval == 250
    assert cfg.eval_iters == 50
    assert cfg.device in ["cuda", "cpu"]


def test_tokenizer_encode_decode():
    text = "hello world minigpt"
    tok = SimpleTokenizer(text)
    assert tok.vocab_size > 0
    encoded = tok.encode("hello")
    decoded = tok.decode(encoded)
    assert decoded == "hello"


def test_minigpt_model_forward():
    cfg = ModelConfig(vocab_size=20, n_embd=32, n_head=2, n_layer=2, block_size=16)
    model = MiniGPT(cfg)
    idx = torch.randint(0, 20, (2, 8))
    logits, loss = model(idx)
    assert logits.shape == (2, 8, 20)
    assert loss is None

    targets = torch.randint(0, 20, (2, 8))
    logits, loss = model(idx, targets)
    assert loss is not None
    assert loss.item() > 0
