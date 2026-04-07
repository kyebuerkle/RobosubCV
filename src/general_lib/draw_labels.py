#	@file: draw_labels.py
#	@brief: draws YOLO bounding boxes onto every image/label pair in a directory
#		and saves the annotated images to an output directory
#
#	Supports two directory layouts:
#		Flat:  input_dir/img.jpg + input_dir/img.txt
#		Split: input_dir/images/img.jpg + input_dir/labels/img.txt
#
#	Usage:
#		python draw_labels.py <input_dir> <output_dir> [--color B G R]
#
#	Examples:
#		python draw_labels.py data/ data/annotated
#		python draw_labels.py data/images data/annotated
#		python draw_labels.py data/images data/annotated --color 0 0 255

import cv2
import numpy as np
import argparse
from pathlib import Path

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def read_label_file(label_file):
	"""
	Reads a YOLO label file and returns a list of dicts with parsed fields.
	Returns an empty list if the file is empty or missing.
	"""
	path = Path(label_file)
	if not path.exists() or path.stat().st_size == 0:
		return []

	boxes = []
	for line in path.read_text().strip().splitlines():
		parts = line.split()
		if len(parts) != 5:
			continue
		class_id = parts[0]
		cx, cy, bw, bh = map(float, parts[1:5])
		boxes.append({"class_id": class_id, "cx": cx, "cy": cy, "bw": bw, "bh": bh})
	return boxes


def imread_unicode(path: Path):
	"""cv2.imread wrapper that handles non-ASCII paths on Windows."""
	buf = np.fromfile(str(path), dtype=np.uint8)
	return cv2.imdecode(buf, cv2.IMREAD_COLOR)


def imwrite_unicode(path: Path, img):
	"""cv2.imwrite wrapper that handles non-ASCII paths on Windows."""
	ext = path.suffix.lower()
	ok, buf = cv2.imencode(ext, img)
	if ok:
		buf.tofile(str(path))
	return ok


def resolve_labels_dir(input_dir: Path):
	"""
	Given the input directory, find where the label .txt files live.

	Supports two layouts:
	  - Split: input_dir/images/ + input_dir/labels/   (pass either root or images/)
	  - Flat:  input_dir/ contains both images and .txt files
	
	Returns (images_dir, labels_dir).
	"""
	# Case 1: user passed the root, which contains images/ and labels/ subfolders
	if (input_dir / "images").is_dir() and (input_dir / "labels").is_dir():
		return input_dir / "images", input_dir / "labels"

	# Case 2: user passed the images/ folder directly; labels/ is a sibling
	sibling_labels = input_dir.parent / "labels"
	if input_dir.name == "images" and sibling_labels.is_dir():
		return input_dir, sibling_labels

	# Case 3: flat layout — images and labels are in the same folder
	return input_dir, input_dir


def draw_labels_on_directory(input_dir, output_dir, color=(0, 255, 0)):
	"""
	Finds every image/label pair and saves a copy of each image
	with its bounding boxes drawn to output_dir.

	:param input_dir: root dir, images/ dir, or flat dir containing images + labels
	:param output_dir: directory to write annotated images to
	:param color: BGR color tuple for the bounding boxes, default green
	"""
	input_dir  = Path(input_dir)
	output_dir = Path(output_dir)
	output_dir.mkdir(parents=True, exist_ok=True)

	images_dir, labels_dir = resolve_labels_dir(input_dir)
	print(f"  images : {images_dir}")
	print(f"  labels : {labels_dir}")

	pairs_found = 0

	for image_file in sorted(images_dir.iterdir()):
		if image_file.suffix.lower() not in IMAGE_EXTENSIONS:
			continue

		label_file = labels_dir / image_file.with_suffix(".txt").name
		if not label_file.exists():
			print(f"  [skip] no label found for {image_file.name}")
			continue

		img = imread_unicode(image_file)
		if img is None:
			print(f"  [skip] could not read image {image_file.name}")
			continue

		h, w = img.shape[:2]
		boxes = read_label_file(label_file)

		for box in boxes:
			cx, cy, bw, bh = box["cx"], box["cy"], box["bw"], box["bh"]
			x1 = int((cx - bw / 2) * w)
			y1 = int((cy - bh / 2) * h)
			x2 = int((cx + bw / 2) * w)
			y2 = int((cy + bh / 2) * h)

			cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
			cv2.putText(img, box["class_id"], (x1, y1 - 6),
			            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

		out_path = output_dir / image_file.name
		imwrite_unicode(out_path, img)
		pairs_found += 1
		print(f"  [saved] {out_path.name}  ({len(boxes)} boxes)")

	print(f"\ndone — {pairs_found} image/label pairs processed")


if __name__ == "__main__":
	parser = argparse.ArgumentParser(
		description="Draw YOLO bounding boxes on all image/label pairs in a directory."
	)
	parser.add_argument("input_dir",  type=str, help="Root dir (with images/ & labels/ subfolders), or images/ dir directly, or a flat dir")
	parser.add_argument("output_dir", type=str, help="Directory to save annotated images")
	parser.add_argument("--color", type=int, nargs=3, default=[0, 255, 0],
	                    metavar=("B", "G", "R"),
	                    help="Bounding box color in BGR (default: 0 255 0 = green)")
	args = parser.parse_args()

	draw_labels_on_directory(args.input_dir, args.output_dir, tuple(args.color))