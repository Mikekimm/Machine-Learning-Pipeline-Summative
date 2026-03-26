from __future__ import annotations

from pathlib import Path

try:
    from src.model import TrainingConfig, train_model
except ModuleNotFoundError:
    from model import TrainingConfig, train_model


def trigger_retraining(project_root: Path) -> dict:
    config = TrainingConfig(
        max_train_per_class=700,
        max_test_per_class=300,
        image_size=(32, 32),
        max_iter=80,
        random_state=42,
    )
    return train_model(project_root=project_root, config=config)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = trigger_retraining(root)
    print("RETRAIN_COMPLETE", result)


if __name__ == "__main__":
    main()
