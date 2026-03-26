from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

try:
    from src.preprocessing import load_split, resolve_default_paths, sample_story_features
except ModuleNotFoundError:
    from preprocessing import load_split, resolve_default_paths, sample_story_features


@dataclass
class TrainingConfig:
    max_train_per_class: int = 500
    max_test_per_class: int = 250
    image_size: tuple[int, int] = (32, 32)
    max_iter: int = 60
    random_state: int = 42


def train_model(project_root: Path, config: TrainingConfig) -> dict:
    paths = resolve_default_paths(project_root)

    x_train, y_train, classes = load_split(
        paths.train_dir,
        max_per_class=config.max_train_per_class,
        image_size=config.image_size,
    )
    x_test, y_test, _ = load_split(
        paths.test_dir,
        max_per_class=config.max_test_per_class,
        image_size=config.image_size,
    )

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=False)),
            (
                "clf",
                LogisticRegression(
                    solver="saga",
                    max_iter=config.max_iter,
                    random_state=config.random_state,
                ),
            ),
        ]
    )

    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    accuracy = float(accuracy_score(y_test, predictions))
    conf = confusion_matrix(y_test, predictions).tolist()
    report = classification_report(y_test, predictions, target_names=classes, output_dict=True)

    model_payload = {
        "model": model,
        "classes": classes,
        "image_size": config.image_size,
    }

    models_dir = project_root / "models"
    results_dir = project_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / "scene_classifier.pkl"
    joblib.dump(model_payload, model_path)

    story_x, story_y = sample_story_features(paths.test_dir, classes, samples_per_class=100)
    story_rows = [
        {
            "class": str(label),
            "brightness": float(feat[0]),
            "blue_ratio": float(feat[1]),
            "green_ratio": float(feat[2]),
            "texture_strength": float(feat[3]),
        }
        for feat, label in zip(story_x, story_y)
    ]

    metrics = {
        "accuracy": accuracy,
        "confusion_matrix": conf,
        "classification_report": report,
        "classes": classes,
        "train_shape": [int(x_train.shape[0]), int(x_train.shape[1])],
        "test_shape": [int(x_test.shape[0]), int(x_test.shape[1])],
        "config": asdict(config),
        "sample_probability_mean": float(np.mean(np.max(probabilities, axis=1))),
    }

    with (results_dir / "metrics.json").open("w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    with (results_dir / "feature_story.json").open("w", encoding="utf-8") as f:
        json.dump(story_rows, f, indent=2)

    return {
        "model_path": str(model_path),
        "metrics_path": str(results_dir / "metrics.json"),
        "story_path": str(results_dir / "feature_story.json"),
        "accuracy": accuracy,
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = train_model(root, TrainingConfig())
    print("TRAINING_COMPLETE", result)


if __name__ == "__main__":
    main()
