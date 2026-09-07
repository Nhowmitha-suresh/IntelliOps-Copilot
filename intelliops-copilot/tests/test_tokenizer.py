import os
import pytest
from tokenizer import BPETokenizer


def test_bpe_roundtrip_edge_cases():
    tokenizer = BPETokenizer()
    sample_text = "abc abc abc hello world 🌍 🚀 repeatedddd"
    tokenizer.train(sample_text, vocab_size=265)

    edge_cases = [
        "",  # Empty string
        "   ",  # Whitespace only
        "\n\t\r",  # Control characters
        "aaaaaa",  # Repeated single character
        "Hello 🌍 🚀! UTF-8 unicode test ✨",  # Emoji & Unicode
        "Python & PyTorch & MiniGPT",  # Special symbols
    ]

    for text in edge_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded)
        assert decoded == text, f"Round-trip failed for: {text!r} (got {decoded!r})"


def test_bpe_vocab_size_and_save_load(tmp_path):
    text = "The quick brown fox jumps over the lazy dog. " * 50
    target_vocab_size = 280

    tokenizer = BPETokenizer()
    tokenizer.train(text, vocab_size=target_vocab_size)

    assert tokenizer.vocab_size == target_vocab_size, f"Vocab size should be {target_vocab_size}"

    save_file = tmp_path / "tokenizer.json"
    tokenizer.save(str(save_file))
    assert os.path.exists(save_file)

    loaded_tokenizer = BPETokenizer.load(str(save_file))
    assert loaded_tokenizer.vocab_size == target_vocab_size

    test_str = "The quick brown fox"
    assert tokenizer.encode(test_str) == loaded_tokenizer.encode(test_str)
    assert loaded_tokenizer.decode(loaded_tokenizer.encode(test_str)) == test_str
