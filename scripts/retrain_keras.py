from __future__ import annotations

import os
import json
from pathlib import Path
import sys

os.environ["MPLBACKEND"] = "Agg"

sys.path.append(str(Path(__file__).resolve().parents[1] / "src"))
from model import TrainingConfig, train_model  # noqa: E402


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = train_model(
        root,
        TrainingConfig(
            image_size=(224, 224),
            batch_size=32,
            head_epochs=6,
            fine_tune_epochs=6,
            unfreeze_last_layers=60,
            learning_rate=1e-3,
            fine_tune_learning_rate=1e-5,
            dropout_rate=0.4,
            dense_units=256,
            validation_split=0.1,
            random_state=42,
        ),
    )
    print(json.dumps(result))


if __name__ == "__main__":
    main()
