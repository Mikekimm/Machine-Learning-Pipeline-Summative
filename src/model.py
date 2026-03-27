from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

try:
    import tensorflow as tf
    from tensorflow.keras import Model, callbacks, layers, optimizers
    from tensorflow.keras.applications.efficientnet import EfficientNetB0, preprocess_input
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    TF_AVAILABLE = True
except Exception:
    tf = None
    Model = object
    callbacks = None
    layers = None
    optimizers = None
    EfficientNetB0 = None
    preprocess_input = None
    ImageDataGenerator = None
    TF_AVAILABLE = False

try:
    import certifi
except Exception:
    certifi = None

try:
    from src.preprocessing import resolve_default_paths, sample_story_features
except ModuleNotFoundError:
    from preprocessing import resolve_default_paths, sample_story_features


@dataclass
class TrainingConfig:
    image_size: tuple[int, int] = (224, 224)
    batch_size: int = 32
    head_epochs: int = 6
    fine_tune_epochs: int = 10
    unfreeze_last_layers: int = 60
    learning_rate: float = 1e-3
    fine_tune_learning_rate: float = 1e-5
    dropout_rate: float = 0.4
    dense_units: int = 256
    validation_split: float = 0.1
    random_state: int = 42


def _build_generators(paths, config: TrainingConfig, class_names: list[str]):
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available in this Python environment. "
            "Use a Python 3.12 venv and install tensorflow before training EfficientNetB0."
        )

    image_size = config.image_size
    target_size = (int(image_size[0]), int(image_size[1]))

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=config.validation_split,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.2,
        horizontal_flip=True,
    )
    eval_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_directory(
        paths.train_dir,
        classes=class_names,
        target_size=target_size,
        batch_size=config.batch_size,
        class_mode="sparse",
        subset="training",
        shuffle=True,
        seed=config.random_state,
    )

    val_gen = train_datagen.flow_from_directory(
        paths.train_dir,
        classes=class_names,
        target_size=target_size,
        batch_size=config.batch_size,
        class_mode="sparse",
        subset="validation",
        shuffle=False,
        seed=config.random_state,
    )

    test_gen = eval_datagen.flow_from_directory(
        paths.test_dir,
        classes=class_names,
        target_size=target_size,
        batch_size=config.batch_size,
        class_mode="sparse",
        shuffle=False,
    )

    return train_gen, val_gen, test_gen


def _build_model(config: TrainingConfig, num_classes: int) -> tuple[Model, Model]:
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available in this Python environment. "
            "Use a Python 3.12 venv and install tensorflow before training EfficientNetB0."
        )

    h, w = int(config.image_size[0]), int(config.image_size[1])
    base = EfficientNetB0(include_top=False, weights="imagenet", input_shape=(h, w, 3))
    base.trainable = False

    inputs = layers.Input(shape=(h, w, 3))
    x = base(inputs, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(config.dense_units, activation="relu")(x)
    x = layers.Dropout(config.dropout_rate)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = Model(inputs, outputs)
    model.compile(
        optimizer=optimizers.Adam(learning_rate=config.learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model, base


def train_model(project_root: Path, config: TrainingConfig) -> dict:
    if not TF_AVAILABLE:
        raise RuntimeError(
            "TensorFlow is not available in this Python environment. "
            "Use a Python 3.12 venv and install tensorflow before training EfficientNetB0."
        )

    if certifi is not None:
        os.environ.setdefault("SSL_CERT_FILE", certifi.where())

    tf.keras.utils.set_random_seed(config.random_state)
    paths = resolve_default_paths(project_root)

    class_names = sorted([d.name for d in paths.train_dir.iterdir() if d.is_dir()])
    train_gen, val_gen, test_gen = _build_generators(paths, config, class_names)

    model, base_model = _build_model(config, num_classes=len(class_names))

    models_dir = project_root / "models"
    results_dir = project_root / "results"
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_path = models_dir / "scene_classifier_best.keras"
    common_callbacks = [
        callbacks.EarlyStopping(monitor="val_accuracy", patience=4, restore_best_weights=True),
        callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.3, patience=2, min_lr=1e-7),
        callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
        ),
    ]

    head_history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.head_epochs,
        callbacks=common_callbacks,
        verbose=1,
    )

    fine_tune_history_dict: dict = {}
    if config.fine_tune_epochs > 0:
        # Fine-tune top layers of EfficientNet for stronger class separation.
        base_model.trainable = True
        freeze_until = max(0, len(base_model.layers) - config.unfreeze_last_layers)
        for layer in base_model.layers[:freeze_until]:
            layer.trainable = False

        model.compile(
            optimizer=optimizers.Adam(learning_rate=config.fine_tune_learning_rate),
            loss="sparse_categorical_crossentropy",
            metrics=["accuracy"],
        )

        fine_tune_history = model.fit(
            train_gen,
            validation_data=val_gen,
            epochs=config.head_epochs + config.fine_tune_epochs,
            initial_epoch=len(head_history.history.get("loss", [])),
            callbacks=common_callbacks,
            verbose=1,
        )
        fine_tune_history_dict = fine_tune_history.history

    if checkpoint_path.exists():
        model = tf.keras.models.load_model(checkpoint_path)

    probabilities = model.predict(test_gen, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    y_true = test_gen.classes

    accuracy = float(accuracy_score(y_true, predictions))
    conf = confusion_matrix(y_true, predictions).tolist()
    report = classification_report(y_true, predictions, target_names=class_names, output_dict=True)

    model_path = models_dir / "scene_classifier.keras"
    model.save(model_path)

    meta = {
        "classes": class_names,
        "image_size": [int(config.image_size[0]), int(config.image_size[1])],
        "architecture": "EfficientNetB0",
    }
    with (models_dir / "scene_classifier_meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    story_x, story_y = sample_story_features(paths.test_dir, class_names, samples_per_class=100)
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
        "precision_weighted": float(report.get("weighted avg", {}).get("precision", 0.0)),
        "recall_weighted": float(report.get("weighted avg", {}).get("recall", 0.0)),
        "f1_weighted": float(report.get("weighted avg", {}).get("f1-score", 0.0)),
        "confusion_matrix": conf,
        "classification_report": report,
        "classes": class_names,
        "train_samples": int(train_gen.samples),
        "val_samples": int(val_gen.samples),
        "test_samples": int(test_gen.samples),
        "config": asdict(config),
        "model_type": "efficientnetb0_transfer_learning",
        "training_history": {
            "head": head_history.history,
            "fine_tune": fine_tune_history_dict,
        },
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
        "precision_weighted": metrics["precision_weighted"],
        "recall_weighted": metrics["recall_weighted"],
        "f1_weighted": metrics["f1_weighted"],
    }


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    result = train_model(root, TrainingConfig())
    print("TRAINING_COMPLETE", result)


if __name__ == "__main__":
    main()
