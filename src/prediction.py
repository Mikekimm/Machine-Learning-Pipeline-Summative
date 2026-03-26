from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

try:
    from src.preprocessing import image_to_model_vector
except ModuleNotFoundError:
    from preprocessing import image_to_model_vector


def load_artifacts(project_root: Path) -> dict:
    model_path = project_root / "models" / "scene_classifier.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    return joblib.load(model_path)


def predict_image(project_root: Path, image_path: Path) -> dict:
    payload = load_artifacts(project_root)
    model = payload["model"]
    classes = payload["classes"]
    image_size = tuple(payload.get("image_size", (32, 32)))

    vector = image_to_model_vector(image_path, image_size=image_size).reshape(1, -1)
    pred_idx = int(model.predict(vector)[0])
    probs = model.predict_proba(vector)[0]

    top_indices = np.argsort(probs)[::-1][:3]
    top_scores = [
        {"class": classes[int(i)], "probability": float(probs[int(i)])} for i in top_indices
    ]

    return {
        "predicted_class": classes[pred_idx],
        "confidence": float(probs[pred_idx]),
        "top_3": top_scores,
    }
