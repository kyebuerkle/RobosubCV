#	@file: run_inference.py
#	@brief: Run a YOLO model on a directory of images, save label .txt files,
#		and optionally draw bounding boxes onto the images.
#
#	Usage:
#		python run_inference.py <input_dir> -m model.pt
#		python run_inference.py <input_dir> <output_dir> -m model.pt
#		python run_inference.py <input_dir> -m model.pt --draw 0 255 0
#		python run_inference.py <input_dir> <output_dir> -m model.pt --draw 255 0 0

import argparse
import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


# ─────────────────────────────────────────────────────────────────
#  Argument parsing
# ─────────────────────────────────────────────────────────────────

def parse_args():
	parser = argparse.ArgumentParser(
		description="Run a YOLO model on a directory of images, save labels, "
		            "and optionally draw bounding boxes.",
		formatter_class=argparse.ArgumentDefaultsHelpFormatter,
	)
	parser.add_argument(
		"input_dir",
		type=str,
		help="Directory of images to run inference on",
	)
	parser.add_argument(
		"output_dir",
		type=str,
		nargs="?",
		default=None,
		help="Directory to save label .txt files (and drawn images if --draw is set). "
		     "Defaults to input_dir.",
	)
	parser.add_argument(
		"-m", "--model",
		required=True,
		metavar="PATH",
		help="Path to YOLO .pt model file",
	)
	parser.add_argument(
		"--draw",
		type=int,
		nargs=3,
		default=None,
		metavar=("R", "G", "B"),
		help="Draw bounding boxes in this RGB color, e.g. --draw 0 255 0 for green. "
		     "Drawn images are saved alongside the label files.",
	)
	parser.add_argument(
		"--conf",
		type=float,
		default=0.25,
		metavar="FLOAT",
		help="Confidence threshold for detections",
	)
	parser.add_argument(
		"--imgsz",
		type=int,
		default=640,
		help="Inference image size",
	)
	parser.add_argument(
		"--device",
		type=str,
		default="cpu",
		help="Inference device: 'cpu', 'cuda', '0', etc.",
	)
	return parser.parse_args()


# ─────────────────────────────────────────────────────────────────
#  Image I/O helpers (from draw_labels.py — handles non-ASCII paths)
# ─────────────────────────────────────────────────────────────────

def imread_unicode(path: Path):
	buf = np.fromfile(str(path), dtype=np.uint8)
	return cv2.imdecode(buf, cv2.IMREAD_COLOR)

def imwrite_unicode(path: Path, img):
	ext = path.suffix.lower()
	ok, buf = cv2.imencode(ext, img)
	if ok:
		buf.tofile(str(path))
	return ok


# ─────────────────────────────────────────────────────────────────
#  Drawing (reuses draw_labels.py logic)
# ─────────────────────────────────────────────────────────────────

def draw_boxes(img, boxes, color_bgr):
	"""Draw YOLO-format boxes [{class_id, cx, cy, bw, bh}] onto img in place."""
	h, w = img.shape[:2]
	for box in boxes:
		cx, cy, bw, bh = box["cx"], box["cy"], box["bw"], box["bh"]
		x1 = int((cx - bw / 2) * w)
		y1 = int((cy - bh / 2) * h)
		x2 = int((cx + bw / 2) * w)
		y2 = int((cy + bh / 2) * h)
		cv2.rectangle(img, (x1, y1), (x2, y2), color_bgr, 2)
		cv2.putText(img, str(box["class_id"]), (x1, y1 - 6),
		            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color_bgr, 1)
	return img


# ─────────────────────────────────────────────────────────────────
#  Main
# ─────────────────────────────────────────────────────────────────

def main():
	args = parse_args()

	input_dir  = Path(args.input_dir).resolve()
	output_dir = Path(args.output_dir).resolve() if args.output_dir else input_dir
	output_dir.mkdir(parents=True, exist_ok=True)

	#	Convert --draw RGB → BGR for OpenCV
	color_bgr = None
	if args.draw is not None:
		r, g, b = args.draw
		color_bgr = (b, g, r)

	print(f"Model:      {args.model}")
	print(f"Input:      {input_dir}")
	print(f"Output:     {output_dir}")
	print(f"Confidence: {args.conf}")
	print(f"Draw:       {args.draw if args.draw else 'off'}")
	print()

	model = YOLO(args.model)

	image_files = sorted(
		f for f in input_dir.iterdir()
		if f.is_file() and f.suffix.lower() in IMAGE_EXTENSIONS
	)

	if not image_files:
		print(f"No images found in {input_dir}")
		return

	processed = 0
	for img_path in image_files:
		results = model.predict(
			source=str(img_path),
			conf=args.conf,
			imgsz=args.imgsz,
			device=args.device,
			verbose=False,
		)

		result  = results[0]
		boxes   = result.boxes
		img_h, img_w = result.orig_shape

		#	Build YOLO-format label lines and box dicts for drawing
		label_lines = []
		box_dicts   = []
		for box in boxes:
			cls_id       = int(box.cls[0])
			cx, cy, bw, bh = box.xywhn[0].tolist()	#	normalised cx, cy, w, h
			label_lines.append(f"{cls_id} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
			box_dicts.append({"class_id": cls_id, "cx": cx, "cy": cy, "bw": bw, "bh": bh})

		#	Save label .txt
		label_path = output_dir / img_path.with_suffix(".txt").name
		label_path.write_text("\n".join(label_lines))

		#	Draw and save annotated image if --draw was passed
		if color_bgr is not None:
			img = imread_unicode(img_path)
			if img is not None:
				img = draw_boxes(img, box_dicts, color_bgr)
				drawn_path = output_dir / img_path.name
				imwrite_unicode(drawn_path, img)

		processed += 1
		print(f"  [done] {img_path.name}  ({len(box_dicts)} detections)")

	print(f"\nDone — {processed} images processed → labels in {output_dir}")


if __name__ == "__main__":
	main()