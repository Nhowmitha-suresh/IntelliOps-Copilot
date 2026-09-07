import os
from minigpt.tokenizer import BPETokenizer


def train_and_evaluate_bpe(
    train_path: str = "minigpt/data/processed/train.txt",
    save_path: str = "minigpt/experiments/tokenizer.json",
    vocab_size: int = 512,
):
    if not os.path.exists(train_path):
        raise FileNotFoundError(f"Training corpus not found at {train_path}")

    print(f"Reading training corpus from {train_path}...")
    with open(train_path, "r", encoding="utf-8") as f:
        train_text = f.read()

    print(f"Training BPETokenizer to target vocab_size={vocab_size} on {len(train_text):,} characters...")
    tokenizer = BPETokenizer()
    tokenizer.train(train_text, vocab_size=vocab_size, verbose=False)

    print(f"Saving tokenizer to {save_path}...")
    tokenizer.save(save_path)

    # Reload tokenizer from JSON to verify save/load
    reloaded_tok = BPETokenizer.load(save_path)

    # 1. Final Vocab Size
    print(f"\nFinal Vocab Size: {reloaded_tok.vocab_size}")

    # 2. 10 Example Learned Merges
    print("\n--- 10 Example Learned Merges ---")
    merges_items = list(reloaded_tok.merges.items())[:10]
    for idx, (pair, new_id) in enumerate(merges_items, 1):
        merged_bytes = reloaded_tok.vocab[new_id]
        print(f"Merge #{idx:2d}: pair {pair} -> token {new_id} ({merged_bytes!r})")

    # 3. Round-trip Sanity Check
    sample_sentence = "First Citizen: Before we proceed any further, hear me speak."
    encoded_ids = reloaded_tok.encode(sample_sentence)
    decoded_sentence = reloaded_tok.decode(encoded_ids)

    print("\n--- Round-Trip Sanity Check ---")
    print(f"Original Text: {sample_sentence!r}")
    print(f"Encoded Token IDs ({len(encoded_ids)} tokens): {encoded_ids}")
    print(f"Decoded Text:  {decoded_sentence!r}")

    exact_match = sample_sentence == decoded_sentence
    print(f"Round-trip Exact Match: {'CONFIRMED [OK]' if exact_match else 'FAILED [FAIL]'}")
    assert exact_match, "Round-trip encode/decode must produce exact original text"



if __name__ == "__main__":
    train_and_evaluate_bpe()
