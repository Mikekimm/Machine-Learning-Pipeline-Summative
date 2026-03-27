from __future__ import annotations

import json
import subprocess
from pathlib import Path

import joblib
import numpy as np

try:
    import tensorflow as tf
    from tensorflow.keras.applications.efficientnet import preprocess_input
    from tensorflow.keras.utils import img_to_array, load_img
    TF_AVAILABLE = True
except Exception:
    tf = None
    preprocess_input = None
    img_to_array = None
    load_img = None
    TF_AVAILABLE = False

try:
    from src.preprocessing import image_to_model_vector
except ModuleNotFoundError:
    from preprocessing import image_to_model_vector


def load_artifacts(project_root: Path) -> dict:
    keras_model_path = project_root / "models" / "scene_classifier.keras"
    keras_meta_path = project_root / "models" / "scene_classifier_meta.json"
    py312_path = project_root / ".venv312" / "bin" / "python"
    predict_script = project_root / "scripts" / "predict_keras.py"

    if TF_AVAILABLE and keras_model_path.exists() and keras_meta_path.exists():
        with keras_meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        model = tf.keras.models.load_model(keras_model_path)
        return {
            "kind": "keras",
            "model": model,
            "classes": meta["classes"],
            "image_size": tuple(meta.get("image_size", [224, 224])),
            "architecture": meta.get("architecture", "EfficientNetB0"),
        }

    if py312_path.exists() and predict_script.exists() and keras_model_path.exists() and keras_meta_path.exists():
        with keras_meta_path.open("r", encoding="utf-8") as f:
            meta = json.load(f)
        return {
            "kind": "keras_subprocess",
            "python": str(py312_path),
            "script": str(predict_script),
            "classes": meta["classes"],
            "image_size": tuple(meta.get("image_size", [224, 224])),
            "architecture": meta.get("architecture", "EfficientNetB0"),
        }

    model_path = project_root / "models" / "scene_classifier.pkl"
    if not model_path.exists():
        raise FileNotFoundError("Model file not found. Train the model first.")
    payload = joblib.load(model_path)
    payload["kind"] = "sklearn"
    return payload


def _predict_with_subprocess(project_root: Path, image_paths: list[Path], payload: dict) -> list[dict]:
    command = [
        payload["python"],
        payload["script"],
        str(project_root),
        *[str(path) for path in image_paths],
    ]
    result = subprocess.run(command, capture_output=True, text=True, check=True)
    return json.loads(result.stdout)


def predict_images(project_root: Path, image_paths: list[Path]) -> list[dict]:
    payload = load_artifacts(project_root)

    if payload.get("kind") == "keras_subprocess":
        return _predict_with_subprocess(project_root, image_paths, payload)

    return [predict_image(project_root, image_path) for image_path in image_paths]


def predict_image(project_root: Path, image_path: Path) -> dict:
    payload = load_artifacts(project_root)
    if payload.get("kind") == "keras_subprocess":
        return _predict_with_subprocess(project_root, [image_path], payload)[0]

    if payload.get("kind") == "keras":
        model = payload["model"]
        classes = payload["classes"]
        image_size = tuple(payload.get("image_size", (224, 224)))

        image = load_img(image_path, target_size=image_size)
        arr = img_to_array(image)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        probs = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        top_indices = np.argsort(probs)[::-1][:3]
        top_scores = [
            {"class": classes[int(i)], "probability": float(probs[int(i)])} for i in top_indices
        ]

        return {
            "predicted_class": classes[pred_idx],
            "confidence": float(probs[pred_idx]),
            "top_3": top_scores,
            "model_type": payload.get("architecture", "EfficientNetB0"),
        }

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
        "model_type": "sklearn_logistic_regression",
    }
