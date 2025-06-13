from dataclasses import dataclass, field

@dataclass
class Config:
    SPECIAL_TOKENS: dict[str, int] = field(default_factory=lambda: {"<pad>": 0, "<eos>": 1, "<unk>": 2})
    NUM_WORKERS: int = 0
    DEVICE: str = "cuda"
