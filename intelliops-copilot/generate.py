import os
import sys
import argparse
import torch
from typing import Optional

from minigpt_config import ModelConfig, ModelConfigVariant
from tokenizer import BPETokenizer, SimpleTokenizer
from model import MiniGPT
from utils import set_seed, load_checkpoint


def load_model_and_tokenizer(
    checkpoint_path: str = "experiments/checkpoints/best.pt",
    tokenizer_path: str = "experiments/tokenizer.json",
    device: str = "cpu",
):
    """
    Loads trained BPETokenizer and MiniGPT model from checkpoint.
    """
    import datetime
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

    if os.path.exists(checkpoint_path):
        ckpt_data = torch.load(checkpoint_path, map_location="cpu")
        state_dict = ckpt_data.get("model_state_dict", ckpt_data.get("model", ckpt_data))
        if "transformer.wte.weight" in state_dict:
            n_embd = state_dict["transformer.wte.weight"].shape[1]
        else:
            n_embd = 128
        if n_embd == 256:
            model_config = ModelConfigVariant(vocab_size=max(tokenizer.vocab_size, 256))
        else:
            model_config = ModelConfig(vocab_size=max(tokenizer.vocab_size, 256))
        model = MiniGPT(model_config)

        mtime = datetime.datetime.fromtimestamp(os.path.getmtime(checkpoint_path))
        size = os.path.getsize(checkpoint_path)
        print(f"Loaded checkpoint: {os.path.abspath(checkpoint_path)}")
        print(f"Checkpoint Modified: {mtime}, Size: {size:,} bytes")
        load_checkpoint(checkpoint_path, model)
    else:
        model_config = ModelConfig(vocab_size=max(tokenizer.vocab_size, 256))
        model = MiniGPT(model_config)
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
    repetition_penalty: float = 1.3,
    repetition_window: int = 100,
    device: str = "cpu",
    seed: Optional[int] = None,
    return_tokens: bool = False,
):
    """
    Autoregressively generates text continuation given a prompt with optional repetition penalty.
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

        # Apply repetition penalty to recent tokens if enabled
        if repetition_penalty != 1.0 and repetition_penalty > 0:
            window = repetition_window if repetition_window > 0 else idx.size(1)
            recent_tokens = idx[0, -window:].tolist()
            unique_recent = set(recent_tokens)
            for token_id in unique_recent:
                if logits[0, token_id] < 0:
                    logits[0, token_id] *= repetition_penalty
                else:
                    logits[0, token_id] /= repetition_penalty

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


def generate_instruction(
    model: MiniGPT,
    tokenizer,
    instruction: str,
    max_new_tokens: int = 100,
    temperature: float = 0.8,
    top_k: Optional[int] = None,
    repetition_penalty: float = 1.3,
    repetition_window: int = 100,
    device: str = "cpu",
    seed: Optional[int] = None,
) -> str:
    """
    Wraps instruction in SFT format:
    ### Instruction:
    {instruction}
    ### Response:
    and returns only the generated response text.
    """
    formatted_prompt = f"### Instruction:\n{instruction}\n### Response:\n"
    full_output = generate_text(
        model=model,
        tokenizer=tokenizer,
        prompt=formatted_prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        repetition_penalty=repetition_penalty,
        repetition_window=repetition_window,
        device=device,
        seed=seed,
    )

    if "### Response:\n" in full_output:
        response_part = full_output.split("### Response:\n", 1)[1]
    elif "### Response:" in full_output:
        response_part = full_output.split("### Response:", 1)[1]
    else:
        response_part = full_output

    if "### Instruction:" in response_part:
        response_part = response_part.split("### Instruction:")[0]

    return response_part.strip()


def main():

    parser = argparse.ArgumentParser(description="Generate text using MiniGPT model.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="experiments/checkpoints/best.pt",
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
    parser.add_argument("--repetition_penalty", type=float, default=1.3, help="Repetition penalty factor")
    parser.add_argument("--repetition_window", type=int, default=100, help="Window size for repetition penalty")
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
    print(f"Repetition Penalty: {args.repetition_penalty} (window={args.repetition_window})")
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
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
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
        repetition_penalty=args.repetition_penalty,
        repetition_window=args.repetition_window,
        device=args.device,
        seed=42,
    )
    print(sampled_text.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(sys.stdout.encoding or "utf-8", errors="replace"))
    print("=" * 50 + "\n")


if __name__ == "__main__":
    main()
