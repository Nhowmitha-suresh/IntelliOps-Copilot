from dataclasses import dataclass, field
import torch


def get_default_device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ModelConfig:
    vocab_size: int = 50257
    n_embd: int = 128
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.1


@dataclass
class ModelConfigVariant:
    vocab_size: int = 50257
    n_embd: int = 256
    n_head: int = 4
    n_layer: int = 4
    block_size: int = 128
    dropout: float = 0.1



@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    max_iters: int = 12000
    eval_interval: int = 500
    eval_iters: int = 50
    device: str = field(default_factory=get_default_device)
