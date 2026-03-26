from pathlib import Path
import random
from PIL import Image

ROOT = Path('archive (14)')
TRAIN = ROOT / 'seg_train' / 'seg_train'
TEST = ROOT / 'seg_test' / 'seg_test'
EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}


def count_images(split_dir: Path) -> dict[str, int]:
    counts: dict[str, int] = {}
    for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        counts[class_dir.name] = sum(
            1 for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in EXTS
        )
    return counts


def sample_open_check(split_dir: Path, samples_per_class: int = 5) -> tuple[bool, int, list[str]]:
    ok = True
    checked = 0
    errors: list[str] = []

    for class_dir in sorted([d for d in split_dir.iterdir() if d.is_dir()]):
        files = [f for f in class_dir.iterdir() if f.is_file() and f.suffix.lower() in EXTS]
        if not files:
            ok = False
            errors.append(f'NO_FILES {split_dir.name}/{class_dir.name}')
            continue

        for file_path in random.sample(files, k=min(samples_per_class, len(files))):
            try:
                with Image.open(file_path) as img:
                    img.verify()
                checked += 1
            except Exception as exc:
                ok = False
                errors.append(f'BAD_IMAGE {file_path}: {exc}')

    return ok, checked, errors


def main() -> None:
    print('TRAIN_EXISTS', TRAIN.exists())
    print('TEST_EXISTS', TEST.exists())

    if not TRAIN.exists() or not TEST.exists():
        raise SystemExit('Expected dataset folders were not found.')

    train_counts = count_images(TRAIN)
    test_counts = count_images(TEST)

    print('TRAIN_COUNTS', train_counts)
    print('TEST_COUNTS', test_counts)

    train_ok, train_checked, train_errors = sample_open_check(TRAIN)
    test_ok, test_checked, test_errors = sample_open_check(TEST)

    print('FILES_CHECKED', train_checked + test_checked)
    for err in train_errors + test_errors:
        print(err)

    print('SANITY_OK', train_ok and test_ok)


if __name__ == '__main__':
    main()
