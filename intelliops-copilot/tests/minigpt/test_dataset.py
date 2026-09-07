import os
import torch
import pytest
from minigpt.dataset import TextDataset


def test_split_ratio():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    clean_path = os.path.join(repo_root, "minigpt", "data", "processed", "corpus_clean.txt")
    train_path = os.path.join(repo_root, "minigpt", "data", "processed", "train.txt")
    val_path = os.path.join(repo_root, "minigpt", "data", "processed", "val.txt")

    assert os.path.exists(clean_path), "Cleaned corpus file must exist"
    assert os.path.exists(train_path), "Train split file must exist"
    assert os.path.exists(val_path), "Val split file must exist"

    with open(clean_path, "r", encoding="utf-8") as f:
        clean_text_str = f.read()
    with open(train_path, "r", encoding="utf-8") as f:
        train_text_str = f.read()
    with open(val_path, "r", encoding="utf-8") as f:
        val_text_str = f.read()

    total_len = len(clean_text_str)
    train_len = len(train_text_str)
    val_len = len(val_text_str)

    assert train_len + val_len == total_len, "Train + Val length must equal total cleaned corpus length"
    expected_train_len = int(total_len * 0.9)
    assert train_len == expected_train_len, f"Train length should be 90% of total: {expected_train_len}"
    assert val_len == total_len - expected_train_len, "Val length should be 10% of total"


def test_text_dataset_shape_and_shift():
    tokens = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80, 90, 100], dtype=torch.long)
    block_size = 4
    dataset = TextDataset(tokens, block_size=block_size)

    assert len(dataset) == 6

    x0, y0 = dataset[0]
    assert isinstance(x0, torch.Tensor)
    assert isinstance(y0, torch.Tensor)
    assert x0.shape == (block_size,)
    assert y0.shape == (block_size,)
    assert torch.equal(x0, torch.tensor([10, 20, 30, 40]))
    assert torch.equal(y0, torch.tensor([20, 30, 40, 50]))

    assert torch.equal(y0[:-1], x0[1:]), "y shifted by 1 relative to x must equal x[1:]"

    x2, y2 = dataset[2]
    assert torch.equal(x2, torch.tensor([30, 40, 50, 60]))
    assert torch.equal(y2, torch.tensor([40, 50, 60, 70]))
    assert torch.equal(y2[:-1], x2[1:])
