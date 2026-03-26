from pathlib import Path
import random
import time

import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('archive (14)')
TRAIN_DIR = ROOT / 'seg_train' / 'seg_train'
TEST_DIR = ROOT / 'seg_test' / 'seg_test'
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
IMG_SIZE = (32, 32)
MAX_TRAIN_PER_CLASS = 400
MAX_TEST_PER_CLASS = 200
SEED = 42


def load_split(split_dir: Path, max_per_class: int):
    classes = sorted([d.name for d in split_dir.iterdir() if d.is_dir()])
    class_to_idx = {c: i for i, c in enumerate(classes)}

    x, y = [], []
    rng = random.Random(SEED)

    for class_name in classes:
        class_dir = split_dir / class_name
        files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
        rng.shuffle(files)
        files = files[:max_per_class]

        for file_path in files:
            with Image.open(file_path) as img:
                img = img.convert('RGB').resize(IMG_SIZE)
                arr = np.asarray(img, dtype=np.float32).reshape(-1) / 255.0
            x.append(arr)
            y.append(class_to_idx[class_name])

    return np.array(x, dtype=np.float32), np.array(y, dtype=np.int64), classes


def main() -> None:
    start = time.time()
    np.random.seed(SEED)

    print('Loading training data...')
    x_train, y_train, classes = load_split(TRAIN_DIR, MAX_TRAIN_PER_CLASS)
    print('Loading test data...')
    x_test, y_test, _ = load_split(TEST_DIR, MAX_TEST_PER_CLASS)

    print('X_train', x_train.shape, 'y_train', y_train.shape)
    print('X_test', x_test.shape, 'y_test', y_test.shape)
    print('Classes', classes)

    # A fast baseline: standardized pixels + multinomial logistic regression.
    model = make_pipeline(
        StandardScaler(with_mean=False),
        LogisticRegression(
            solver='saga',
            max_iter=30,
            random_state=SEED,
            n_jobs=-1,
            verbose=0,
        ),
    )

    print('Training baseline model...')
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)

    print('BASELINE_TEST_ACCURACY', round(acc, 4))
    print('CLASS_REPORT')
    print(classification_report(y_test, preds, target_names=classes, digits=3))
    print('ELAPSED_SECONDS', round(time.time() - start, 2))


if __name__ == '__main__':
    main()
