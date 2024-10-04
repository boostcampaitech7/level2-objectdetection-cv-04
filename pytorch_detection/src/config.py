from dataclasses import dataclass

@dataclass
class Config:
    batch_size: int = 16
    num_workers: int = 8
    num_classes: int = 11
    lr: float = 0.005
    momentum: float = 0.9
    weight_decay: float = 0.0005
    num_epochs: int = 12

def get_config():
    return Config()