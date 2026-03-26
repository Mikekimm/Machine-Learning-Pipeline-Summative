from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from PIL import Image

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


@dataclass
class DatasetPaths:
    train_dir: Path
    test_dir: Path


def resolve_default_paths(project_root: Path) -> DatasetPaths:
    train_dir = project_root / "archive (14)" / "seg_train" / "seg_train"
    test_dir = project_root / "archive (14)" / "seg_test" / "seg_test"
    return DatasetPaths(train_dir=train_dir, test_dir=test_dir)


def list_images(folder: Path) -> list[Path]:
    return [
        p
        for p in sorted(folder.iterdir())
        if p.is_file() and p.suffix.lower() in VALID_EXTENSIONS
    ]


def image_to_model_vector(image_path: Path, image_size: tuple[int, int] = (32, 32)) -> np.ndarray:
    with Image.open(image_path) as img:
        resized = img.convert("RGB").resize(image_size)
        arr = np.asarray(resized, dtype=np.float32)
    return (arr.reshape(-1) / 255.0).astype(np.float32)


def image_to_story_features(image_path: Path, image_size: tuple[int, int] = (64, 64)) -> np.ndarray:
    with Image.open(image_path) as img:
        resized = img.convert("RGB").resize(image_size)
        arr = np.asarray(resized, dtype=np.float32)

    brightness = arr.mean() / 255.0
    blue_ratio = arr[..., 2].mean() / (arr.mean() + 1e-6)
    green_ratio = arr[..., 1].mean() / (arr.mean() + 1e-6)

    gray = arr.mean(axis=2)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    texture_strength = float(np.mean(np.abs(gx)) + np.mean(np.abs(gy))) / 255.0

    return np.array([brightness, blue_ratio, green_ratio, texture_strength], dtype=np.float32)


def load_split(
    split_dir: Path,
    max_per_class: int | None = None,
    image_size: tuple[int, int] = (32, 32),
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {name: idx for idx, name in enumerate(classes)}

    x: list[np.ndarray] = []
    y: list[int] = []

    for class_name in classes:
        class_dir = split_dir / class_name
        files = list_images(class_dir)
        if max_per_class is not None:
            files = files[:max_per_class]

        for image_path in files:
            x.append(image_to_model_vector(image_path, image_size=image_size))
            y.append(class_to_idx[class_name])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int64), classes


def sample_story_features(
    split_dir: Path,
    classes: Iterable[str],
    samples_per_class: int = 80,
) -> tuple[np.ndarray, np.ndarray]:
    features: list[np.ndarray] = []
    labels: list[str] = []

    for class_name in classes:
        class_dir = split_dir / class_name
        files = list_images(class_dir)[:samples_per_class]
        for image_path in files:
            features.append(image_to_story_features(image_path))
            labels.append(class_name)

    return np.array(features, dtype=np.float32), np.array(labels)
