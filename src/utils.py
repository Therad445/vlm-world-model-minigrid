from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from PIL import Image


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_device(name: str) -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def ensure_parent(path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def resize_frame(frame: np.ndarray, size: int) -> np.ndarray:
    image = Image.fromarray(frame).resize((size, size), Image.Resampling.NEAREST)
    return np.asarray(image, dtype=np.uint8)


def obs_to_tensor(frame: np.ndarray, device: torch.device) -> torch.Tensor:
    return (
        torch.from_numpy(frame.copy())
        .permute(2, 0, 1)
        .unsqueeze(0)
        .to(device=device, dtype=torch.float32)
        / 255.0
    )

