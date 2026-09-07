import os
import sys
import argparse
import torch
from typing import Optional

from minigpt_config import ModelConfig
from tokenizer import BPETokenizer, SimpleTokenizer
from model import MiniGPT
from utils import set_seed, load_checkpoint


def load_model_and_tokenizer(
    checkpoint_path: str = "experiments/checkpoints/best_model.pt",
    tokenizer_path: str = "experiments/tokenizer.json",
    device: str = "cpu",
):
    """
    Loads trained BPETokenizer and MiniGPT model from checkpoint.
    """
    # Map checkpoint alias if needed
    if not os.path.exists(checkpoint_path):
        alt_path = checkpoint_path.replace("best.pt", "best_model.pt")
        if os.path.exists(alt_path):
            checkpoint_path = alt_path

    # Load tokenizer
    if os.path.exists(tokenizer_path):
        tokenizer = BPETokenizer.load(tokenizer_path)
    else:
        tokenizer = SimpleTokenizer("MiniGPT default text prompt")

    model_config = ModelConfig(vocab_size=max(tokenizer.vocab_size, 256))
    model = MiniGPT(model_config)

    if os.path.exists(checkpoint_path):
        load_checkpoint(checkpoint_path, model)
    else:
        print(f"Warning: Checkpoint not found at {checkpoint_path}, using randomly initialized model.")

    model.to(device)
    model.eval()
    return model, tokenizer


def generate_text(
    model: MiniGPT,
    tokenizer,
    prompt: str = "Once upon a time",
    max_new_tokens: int = 200,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    device: str = "cpu",
    seed: Optional[int] = None,
    return_tokens: bool = False,
):
    """
    Autoregressively generates text continuation given a prompt.
    """
    if seed is not None:
        set_seed(seed)

    model.eval()
    model.to(device)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [0]

    idx = torch.tensor([tokens], dtype=torch.long, device=device)

    for _ in range(max_new_tokens):
        # Sliding window if sequence length exceeds block_size
        idx_cond = idx if idx.size(1) <= model.config.block_size else idx[:, -model.config.block_size :]

        with torch.no_grad():
            logits, _ = model(idx_cond)

        logits = logits[:, -1, :]  # shape: (1, vocab_size)

        if temperature <= 1e-5:
            # Greedy sampling
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)
        else:
            logits = logits / temperature
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float("Inf")

            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)

        idx = torch.cat((idx, idx_next), dim=1)

    generated_tokens = idx[0].tolist()
    decoded_text = tokenizer.decode(generated_tokens)
    if return_tokens:
        return decoded_text, generated_tokens
    return decoded_text


def main():
    parser = argparse.ArgumentParser(description="Generate text using MiniGPT model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--tokenizer",
        type=str,
        default="experiments/tokenizer.json",
        help="Path to tokenizer file",
    )
    parser.add_argument("--prompt", type=str, default="Once upon a time", help="Text prompt")
    parser.add_argument("--max_new_tokens", type=int, default=200, help="Number of new tokens to generate")
    parser.add_argument("--temperature", type=float, default=0.8, help="Sampling temperature")
    parser.add_argument("--top_k", type=int, default=None, help="Top-k sampling threshold")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")

    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.checkpoint, args.tokenizer, args.device)

    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass

    print("\n" + "=" * 50)
    print(f"Prompt: {args.prompt!r}")
    print(f"Checkpoint: {args.checkpoint}")
    print("=" * 50)

    # 1. Greedy Generation (temperature near-zero)
    print("\n--- 1. Greedy Output (temperature = 1e-5) ---")
    greedy_text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=1e-5,
        top_k=None,
        device=args.device,
        seed=42,
    )
    print(greedy_text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))

    # 2. Sampled Generation (temperature > 0)
    print(f"\n--- 2. Sampled Output (temperature = {args.temperature}, top_k = {args.top_k}) ---")
    sampled_text = generate_text(
        model,
        tokenizer,
        prompt=args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        device=args.device,
        seed=42,
    )
    print(sampled_text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
