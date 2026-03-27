from __future__ import annotations

import json
import subprocess
from pathlib import Path

try:
    from src.model import TrainingConfig, train_model
except ModuleNotFoundError:
    from model import TrainingConfig, train_model


def trigger_retraining(project_root: Path) -> dict:
    py312 = project_root / ".venv312" / "bin" / "python"
    script = project_root / "scripts" / "retrain_keras.py"

    if py312.exists() and script.exists():
        result = subprocess.run(
            [str(py312), str(script)],
            capture_output=True,
            text=True,
            check=True,
        )
        return json.loads(result.stdout)

    config = TrainingConfig(
        image_size=(224, 224),
        batch_size=32,
        head_epochs=6,
        fine_tune_epochs=10,
        unfreeze_last_layers=60,
        learning_rate=1e-3,
        fine_tune_learning_rate=1e-5,
        dropout_rate=0.4,
        dense_units=256,
        validation_split=0.1,
        random_state=42,
    )
    return train_model(project_root=project_root, config=config)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = trigger_retraining(root)
    print("RETRAIN_COMPLETE", result)


if __name__ == "__main__":
    main()
