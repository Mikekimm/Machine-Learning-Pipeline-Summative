from __future__ import annotations

import os
import json
import sys
from pathlib import Path

os.environ["MPLBACKEND"] = "Agg"

import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.utils import img_to_array, load_img


def main() -> None:
    if len(sys.argv) < 3:
        raise SystemExit("Usage: predict_keras.py <project_root> <image1> [image2 ...]")

    project_root = Path(sys.argv[1]).resolve()
    image_paths = [Path(p).resolve() for p in sys.argv[2:]]

    model_path = project_root / "models" / "scene_classifier.keras"
    meta_path = project_root / "models" / "scene_classifier_meta.json"

    if not model_path.exists() or not meta_path.exists():
        raise FileNotFoundError("Keras model artifacts are missing")

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    classes = meta["classes"]
    image_size = tuple(meta.get("image_size", [224, 224]))

    model = tf.keras.models.load_model(model_path)
    out = []

    for image_path in image_paths:
        image = load_img(image_path, target_size=image_size)
        arr = img_to_array(image)
        arr = np.expand_dims(arr, axis=0)
        arr = preprocess_input(arr)

        probs = model.predict(arr, verbose=0)[0]
        pred_idx = int(np.argmax(probs))

        top_indices = np.argsort(probs)[::-1][:3]
        top_scores = [
            {"class": classes[int(i)], "probability": float(probs[int(i)])}
            for i in top_indices
        ]

        out.append(
            {
                "image": image_path.name,
                "predicted_class": classes[pred_idx],
                "confidence": float(probs[pred_idx]),
                "top_3": top_scores,
                "model_type": meta.get("architecture", "EfficientNetB0"),
            }
        )

    print(json.dumps(out))


if __name__ == "__main__":
    main()
