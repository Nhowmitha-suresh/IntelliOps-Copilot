import torch
import pytest
from minigpt_config import ModelConfig
from tokenizer import SimpleTokenizer, BPETokenizer
from model import MiniGPT
from generate import generate_text


def test_generate_output_length():
    cfg = ModelConfig(vocab_size=260, block_size=16, n_embd=32, n_head=2, n_layer=1)
    model = MiniGPT(cfg)
    tokenizer = BPETokenizer()
    tokenizer.train("Hello world MiniGPT token generator test string", vocab_size=260)

    prompt = "Hello"
    prompt_tokens = tokenizer.encode(prompt)
    max_new = 15

    out_text, out_tokens = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_new, temperature=0.8, return_tokens=True)

    # Generated token sequence length must equal len(prompt_tokens) + max_new
    assert len(out_tokens) == len(prompt_tokens) + max_new


def test_top_k_restricts_sampling():
    cfg = ModelConfig(vocab_size=260, block_size=16, n_embd=32, n_head=2, n_layer=1)
    model = MiniGPT(cfg)
    tokenizer = BPETokenizer()
    tokenizer.train("Hello world MiniGPT token generator test string", vocab_size=260)

    prompt = "Hello"
    max_new = 10

    # With top_k=1, sampling is restricted to only the highest probability token (greedy equivalence)
    top1_out = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_new, temperature=0.8, top_k=1, seed=42)
    greedy_out = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_new, temperature=1e-5, seed=42)

    assert top1_out == greedy_out, "top_k=1 must produce identical output to greedy decoding"


def test_near_zero_temperature_deterministic():
    cfg = ModelConfig(vocab_size=260, block_size=16, n_embd=32, n_head=2, n_layer=1)
    model = MiniGPT(cfg)
    tokenizer = BPETokenizer()
    tokenizer.train("Hello world MiniGPT token generator test string", vocab_size=260)

    prompt = "MiniGPT prompt"
    max_new = 12

    out1 = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_new, temperature=1e-5, seed=123)
    out2 = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_new, temperature=1e-5, seed=123)
    out3 = generate_text(model, tokenizer, prompt=prompt, max_new_tokens=max_new, temperature=1e-5, seed=999)

    assert out1 == out2, "Greedy generation with same seed must be 100% deterministic"
    assert out1 == out3, "Greedy generation (argmax) must be deterministic regardless of seed"
