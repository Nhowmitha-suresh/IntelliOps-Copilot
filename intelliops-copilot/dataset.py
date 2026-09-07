from typing import Union, List, Tuple
import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):
    """
    PyTorch Dataset for MiniGPT language model training.
    Takes a tokenized sequence tensor/list and block_size.
    Returns (x, y) next-token-prediction pairs via __getitem__,
    where x is input sequence of length block_size, and y is target sequence shifted by 1 position.
    """

    def __init__(self, data: Union[torch.Tensor, List[int]], block_size: int):
        if isinstance(data, torch.Tensor):
            self.data = data.detach().cpu().to(dtype=torch.long)
        else:
            self.data = torch.tensor(data, dtype=torch.long)
        self.block_size = block_size

    def __len__(self) -> int:
        if len(self.data) <= self.block_size:
            return 0
        return len(self.data) - self.block_size

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.data[idx : idx + self.block_size]
        y = self.data[idx + 1 : idx + 1 + self.block_size]
        return x, y
