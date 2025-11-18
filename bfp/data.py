from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader


@dataclass
class LogDataConfig:
    path: Path
    sequence_length: int = 60
    label_column: str = "label"


class LogDataset(Dataset):
    """
    CSV 예시:
        timestamp, f1, f2, ..., label
    """

    def __init__(self, config: LogDataConfig):
        self.config = config
        df = pd.read_csv(config.path)

        if "timestamp" in df.columns:
            df = df.sort_values("timestamp")

        if config.label_column not in df.columns:
            raise ValueError(f"label column '{config.label_column}' not found")

        self.labels = df[config.label_column].astype(np.float32)
        features = df.drop(columns=[config.label_column])

        if "timestamp" in features.columns:
            features = features.drop(columns=["timestamp"])

        self.features_array = features.to_numpy(dtype=np.float32)
        self.labels_array = self.labels.to_numpy(dtype=np.float32)
        self.sequence_length = config.sequence_length

        if len(self.features_array) <= self.sequence_length:
            raise ValueError("not enough rows for sequence window")

    def __len__(self) -> int:
        return len(self.features_array) - self.sequence_length

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        x = self.features_array[idx : idx + self.sequence_length]
        y = self.labels_array[idx + self.sequence_length - 1]
        return torch.from_numpy(x), torch.tensor(y)


def create_dataloaders(
    train_path: Path,
    valid_path: Path | None,
    sequence_length: int,
    batch_size: int,
    num_workers: int = 4,
):
    train_ds = LogDataset(LogDataConfig(path=train_path, sequence_length=sequence_length))
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )

    valid_loader = None
    if valid_path is not None and valid_path.exists():
        valid_ds = LogDataset(LogDataConfig(path=valid_path, sequence_length=sequence_length))
        valid_loader = DataLoader(
            valid_ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=False,
        )

    return train_loader, valid_loader
