import pytest
from pathlib import Path
from PIL import Image

SPLITS = ["train", "valid", "test"]


@pytest.mark.parametrize("split", SPLITS)
def test_split_structure(dataset_root, split):
    split_dir = dataset_root / split
    if not split_dir.exists():
        pytest.skip(f"{split} split not present")

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    assert images_dir.is_dir(), f"{split}/images missing"
    assert labels_dir.is_dir(), f"{split}/labels missing"


@pytest.mark.parametrize("split", SPLITS)
def test_images_are_valid(dataset_root, split):
    split_dir = dataset_root / split
    if not split_dir.exists():
        pytest.skip(f"{split} split not present")

    images_dir = split_dir / "images"
    images = [p for p in images_dir.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    assert images, f"No images found in {images_dir}"

    for img_path in images:
        with Image.open(img_path) as img:
            img.verify()
            assert img.format in {"JPEG", "PNG"}


@pytest.mark.parametrize("split", SPLITS)
def test_labels_exist_for_each_image(dataset_root, split):
    split_dir = dataset_root / split
    if not split_dir.exists():
        pytest.skip(f"{split} split not present")

    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"

    images = [p for p in images_dir.iterdir() if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

    assert images, f"No images in {images_dir}"

    for img_path in images:
        label_file = labels_dir / f"{img_path.stem}.txt"
        assert label_file.is_file(), f"Missing label file for {img_path.name}"


@pytest.mark.parametrize("split", SPLITS)
def test_labels_format(dataset_root, split):
    split_dir = dataset_root / split
    if not split_dir.exists():
        pytest.skip(f"{split} split not present")

    labels_dir = split_dir / "labels"
    if not labels_dir.exists():
        pytest.skip(f"{split} labels directory missing")

    for label_path in labels_dir.iterdir():
        if label_path.suffix.lower() != ".txt":
            continue

        with open(label_path) as f:
            lines = f.readlines()

        for line in lines:
            parts = line.strip().split()
            if len(parts) > 5:
                print(f"Skipping test for Polygon label in file: {label_path}")
                continue
            assert len(parts) == 5, f"Incorrect label format in {label_path}"
            class_id, x, y, w, h = parts
            # Check types
            try:
                class_id_int = int(class_id)
                x = float(x)
                y = float(y)
                w = float(w)
                h = float(h)
            except ValueError:
                pytest.fail(f"Non-numeric label in {label_path}: {line}")

            # Check normalized coordinates
            for val in [x, y, w, h]:
                assert 0.0 <= val <= 1.0, f"Value out of range in {label_path}: {line}"
