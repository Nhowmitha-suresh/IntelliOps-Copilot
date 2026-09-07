import json
import os
from typing import List, Dict, Tuple, Optional


def get_stats(ids: List[int]) -> Dict[Tuple[int, int], int]:
    """Counts frequency of adjacent pairs in an ID sequence."""
    counts: Dict[Tuple[int, int], int] = {}
    for p0, p1 in zip(ids, ids[1:]):
        pair = (p0, p1)
        counts[pair] = counts.get(pair, 0) + 1
    return counts


def merge_pair(ids: List[int], pair: Tuple[int, int], new_id: int) -> List[int]:
    """Replaces all occurrences of pair (p0, p1) with new_id in sequence ids."""
    new_ids = []
    i = 0
    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i + 1] == pair[1]:
            new_ids.append(new_id)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1
    return new_ids


class BPETokenizer:
    """
    From-scratch Byte-Pair Encoding (BPE) Tokenizer.
    Operates at the UTF-8 byte level, guaranteeing OOV-free handling of any text.
    """

    def __init__(self):
        # Base vocabulary for single bytes 0..255
        self.vocab: Dict[int, bytes] = {i: bytes([i]) for i in range(256)}
        self.merges: Dict[Tuple[int, int], int] = {}

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """
        Iteratively merges the most frequent adjacent pair of byte tokens until target vocab_size is reached.
        """
        if vocab_size < 256:
            raise ValueError("vocab_size must be at least 256 for byte-level BPE")

        ids = list(text.encode("utf-8"))
        num_merges = vocab_size - 256

        for i in range(num_merges):
            stats = get_stats(ids)
            if not stats:
                break

            # Pick pair with maximum frequency
            best_pair = max(stats, key=stats.get)
            if stats[best_pair] < 2 and len(ids) > 1:
                # If highest frequency is < 2, stop merging unless user explicitly requested
                if stats[best_pair] < 1:
                    break

            new_id = 256 + i
            ids = merge_pair(ids, best_pair, new_id)

            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]

            if verbose and (i % 50 == 0 or i == num_merges - 1):
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_id} ({self.vocab[new_id]!r})")

    def encode(self, text: str) -> List[int]:
        """Encodes string text into a list of BPE token IDs."""
        if not text:
            return []

        ids = list(text.encode("utf-8"))

        while len(ids) >= 2:
            stats = get_stats(ids)
            # Find the pair that was merged earliest (lowest new_id)
            pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
            if pair not in self.merges:
                break
            new_id = self.merges[pair]
            ids = merge_pair(ids, pair, new_id)

        return ids

    def decode(self, token_ids: List[int]) -> str:
        """Decodes list of BPE token IDs back into string text."""
        if not token_ids:
            return ""

        byte_parts = []
        for token in token_ids:
            if token in self.vocab:
                byte_parts.append(self.vocab[token])
            else:
                # Fallback for unexpected tokens
                byte_parts.append(bytes([token % 256]))

        raw_bytes = b"".join(byte_parts)
        return raw_bytes.decode("utf-8", errors="replace")

    def save(self, path: str) -> None:
        """Saves BPE vocabulary and merge rules to a JSON file."""
        os.makedirs(os.path.dirname(path), exist_ok=True)
        merges_serialized = [[[p0, p1], new_id] for (p0, p1), new_id in self.merges.items()]
        data = {
            "vocab_size": self.vocab_size,
            "merges": merges_serialized,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "BPETokenizer":
        """Loads BPETokenizer from a JSON file."""
        tokenizer = cls()
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        for (p0, p1), new_id in data.get("merges", []):
            pair = (int(p0), int(p1))
            new_id = int(new_id)
            tokenizer.merges[pair] = new_id
            tokenizer.vocab[new_id] = tokenizer.vocab[pair[0]] + tokenizer.vocab[pair[1]]

        return tokenizer


class SimpleTokenizer:
    """
    A standalone character-level tokenizer for MiniGPT.
    Fully isolated from pre-trained tokenizers or app dependencies.
    """

    def __init__(self, text: str = None):
        self.stoi: Dict[str, int] = {}
        self.itos: Dict[int, str] = {}
        if text is not None:
            self.fit(text)

    def fit(self, text: str) -> None:
        chars = sorted(list(set(text)))
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for i, ch in enumerate(chars)}

    @property
    def vocab_size(self) -> int:
        return len(self.stoi)

    def encode(self, text: str) -> List[int]:
        return [self.stoi[c] for c in text if c in self.stoi]

    def decode(self, tokens: List[int]) -> str:
        return "".join([self.itos[i] for i in tokens if i in self.itos])
