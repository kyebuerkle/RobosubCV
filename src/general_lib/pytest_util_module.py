#	@file: pytest_util_module.py
# 	@brief: this scripts holds util functions for pytests, like dataset validation

from pathlib import Path
from PIL import Image
import pytest

SPLITS = ["train", "valid", "test"]

def validate_yolov8_dataset(dataset_root: Path):
	for split in SPLITS:
		split_dir = dataset_root / split
		if not split_dir.exists():
			continue

		images_dir = split_dir / "images"
		labels_dir = split_dir / "labels"

		assert images_dir.is_dir(), f"{split}/images missing"
		assert labels_dir.is_dir(), f"{split}/labels missing"

		images = [p for p in images_dir.iterdir()
				if p.suffix.lower() in {".jpg", ".jpeg", ".png"}]

		assert images, f"No images found in {images_dir}"

		for img_path in images:
			# Validate image
			with Image.open(img_path) as img:
				img.verify()
				assert img.format in {"JPEG", "PNG"}

			# Validate label existence
			label_file = labels_dir / f"{img_path.stem}.txt"
			assert label_file.is_file(), f"Missing label for {img_path.name}"

		# Validate label format
		for label_path in labels_dir.glob("*.txt"):
			with open(label_path) as f:
				lines = f.readlines()

			for line in lines:
				parts = line.strip().split()
				if len(parts) > 5:
					continue  # polygon case

				assert len(parts) == 5, f"Incorrect label format in {label_path}"

				class_id, x, y, w, h = parts

				int(class_id)
				for val in map(float, [x, y, w, h]):
					assert 0.0 <= val <= 1.0
