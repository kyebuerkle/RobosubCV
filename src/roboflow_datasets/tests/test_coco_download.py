import json
from pathlib import Path
from PIL import Image
import pytest

SPLITS = ["train", "valid", "test"]


@pytest.mark.parametrize("split", SPLITS)
def test_split_structure(dataset_root, split):
    split_dir = dataset_root / split
    assert (split_dir).is_dir()
    assert (split_dir / "_annotations.coco.json").is_file()


@pytest.mark.parametrize("split", SPLITS)
def test_images_are_valid(dataset_root, split):
    split_dir = dataset_root / split
    if not split_dir.exists():
        pytest.skip(f"{split} split not present")

    images = [
		p for p in split_dir.glob("*")
		if p.suffix.lower() not in {".json", ".txt"}
		]


    assert images, f"No images found in {split_dir}"
    for img_path in images:
        with Image.open(img_path) as img:
            img.verify()
            assert img.format in {"JPEG", "PNG"}


@pytest.mark.parametrize("split", SPLITS)
def test_coco_annotations_valid(dataset_root, split):
    split_dir = dataset_root / split
    if not split_dir.exists():
        pytest.skip(f"{split} split not present")

    with open(split_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    for key in ("images", "annotations", "categories"):
        assert key in coco


@pytest.mark.parametrize("split", SPLITS)
def test_annotations_reference_real_images(dataset_root, split):
    split_dir = dataset_root / split
    if not split_dir.exists():
        pytest.skip(f"{split} split not present")

    with open(split_dir / "_annotations.coco.json") as f:
        coco = json.load(f)

    image_files = {p.name for p in split_dir.iterdir()}
    image_ids = {img["id"]: img["file_name"] for img in coco["images"]}

    for file_name in image_ids.values():
        assert file_name in image_files

    for ann in coco["annotations"]:
        assert ann["image_id"] in image_ids
