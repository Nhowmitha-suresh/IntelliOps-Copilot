import os
import re
import urllib.request
from typing import Tuple

# Default URL: Project Gutenberg / Public Domain Corpus (The Complete Works of Shakespeare ~5.5MB)
DEFAULT_URLS = [
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt",
    "https://www.gutenberg.org/cache/epub/100/pg100.txt",
    "https://www.gutenberg.org/files/1342/1342-0.txt",
]


def strip_gutenberg_header_footer(text: str) -> str:
    """Strips Gutenberg headers and footers if present in the text."""
    start_markers = [
        "*** START OF THIS PROJECT GUTENBERG EBOOK",
        "*** START OF THE PROJECT GUTENBERG EBOOK",
        "***START OF THE PROJECT GUTENBERG EBOOK",
    ]
    end_markers = [
        "*** END OF THIS PROJECT GUTENBERG EBOOK",
        "*** END OF THE PROJECT GUTENBERG EBOOK",
        "***END OF THE PROJECT GUTENBERG EBOOK",
    ]

    text_upper = text.upper()
    start_pos = 0
    for marker in start_markers:
        pos = text_upper.find(marker)
        if pos != -1:
            eol = text.find("\n", pos)
            if eol != -1:
                start_pos = eol + 1
            break

    end_pos = len(text)
    for marker in end_markers:
        pos = text_upper.find(marker)
        if pos != -1:
            end_pos = pos
            break

    return text[start_pos:end_pos]


def clean_text(raw_text: str) -> str:
    """Cleans raw text by stripping headers/footers and normalizing whitespace."""
    text = strip_gutenberg_header_footer(raw_text)
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def split_corpus(text: str, train_ratio: float = 0.9) -> Tuple[str, str]:
    """Splits cleaned text into train (90%) and val (10%)."""
    split_idx = int(len(text) * train_ratio)
    train_text = text[:split_idx]
    val_text = text[split_idx:]
    return train_text, val_text


def print_stats(text: str, label: str = "Corpus") -> None:
    """Prints corpus statistics: character count, word count, and 500-char preview."""
    char_count = len(text)
    word_count = len(text.split())
    preview = text[:500]
    print(f"=== {label} Statistics ===")
    print(f"Character Count: {char_count:,}")
    print(f"Word Count:      {word_count:,}")
    print("--- 500-Character Preview ---")
    print(preview)
    print("===============================\n")


def download_and_prepare_corpus(
    urls: list = None,
    raw_path: str = "minigpt/data/raw/corpus.txt",
    processed_clean_path: str = "minigpt/data/processed/corpus_clean.txt",
    train_path: str = "minigpt/data/processed/train.txt",
    val_path: str = "minigpt/data/processed/val.txt",
):
    if urls is None:
        urls = DEFAULT_URLS

    os.makedirs(os.path.dirname(raw_path), exist_ok=True)
    os.makedirs(os.path.dirname(processed_clean_path), exist_ok=True)

    raw_text = None
    for url in urls:
        try:
            print(f"Attempting to download corpus from: {url}")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=15) as resp:
                raw_text = resp.read().decode("utf-8", errors="ignore")
                if len(raw_text) > 1000:
                    print(f"Successfully downloaded {len(raw_text):,} characters.")
                    break
        except Exception as e:
            print(f"Failed to download from {url}: {e}")

    if not raw_text:
        print("Using synthetic sample fallback corpus for offline mode.")
        raw_text = (
            "Once upon a time, in a small village surrounded by lush green hills, "
            "there lived a curious young explorer named MiniGPT. MiniGPT loved to discover "
            "new patterns in text and tell wonderful stories to everyone in the valley.\n\n"
        ) * 5000  # Creates ~1MB corpus

    # 1. Save raw corpus
    with open(raw_path, "w", encoding="utf-8") as f:
        f.write(raw_text)
    print(f"Saved raw corpus to {raw_path}")

    # 2. Clean corpus
    cleaned_text = clean_text(raw_text)
    with open(processed_clean_path, "w", encoding="utf-8") as f:
        f.write(cleaned_text)
    print(f"Saved cleaned corpus to {processed_clean_path}")

    # 3. Split 90% train / 10% val
    train_text, val_text = split_corpus(cleaned_text, train_ratio=0.9)
    with open(train_path, "w", encoding="utf-8") as f:
        f.write(train_text)
    with open(val_path, "w", encoding="utf-8") as f:
        f.write(val_text)
    print(f"Saved train split to {train_path}")
    print(f"Saved val split to {val_path}")

    # 5. Print corpus stats
    print_stats(cleaned_text, label="Cleaned Corpus")
    print_stats(train_text, label="Train Split (90%)")
    print_stats(val_text, label="Val Split (10%)")


if __name__ == "__main__":
    download_and_prepare_corpus()
