#	@file: geometric_module.py
#	@brief: this script holds geometric augmentation functions
#		Eg. change_scale

import cv2
import numpy as np
from pathlib import Path

AREA_RATIO_MIN = 0.5
PX_MIN_W = 5
PX_MIN_H = 10
PX_MAX_W = 15000
PX_MAX_H = 15000

def change_scale(image_file, out_file, scale_amount: float, origin: tuple[float, float] | None = None):
	"""
	changes the scale of an image

	:param image_file: path to image
	:param out_file: path to output scaled image
	:param scale_amount: amount to scale image
	:type scale_amount: float
	:param origin: the origin point that the image scales from: (x, y), none = center
	:type origin: tuple ( float, float )
	"""
	img = cv2.imread(str(image_file))
	if img is None:
		raise FileNotFoundError(f"Could not read image: {image_file}")

	if scale_amount == 1.0:
		cv2.imwrite(str(out_file), img)
		return

	h, w = img.shape[:2]
	ox, oy = origin if origin is not None else (w / 2.0, h / 2.0)

	# Build 2×3 affine matrix for: translate(-o) → scale → translate(+o)
	# [ scale   0      ox*(1-scale) ]
	# [ 0       scale  oy*(1-scale) ]
	M = np.array(
		[
			[scale_amount, 0, ox * (1 - scale_amount)],
			[0, scale_amount, oy * (1 - scale_amount)],
		], dtype = np.float64)

	interp = cv2.INTER_LINEAR if scale_amount > 1.0 else cv2.INTER_AREA
	result = cv2.warpAffine(
		img, M, (w, h),
		flags=interp,
		borderMode=cv2.BORDER_CONSTANT,
		borderValue=0,
		)

	cv2.imwrite(str(out_file), result)
	return out_file

# each line is: id x y w h
def yolo_scale_label(label_file, out_file, scale_amount: float, image_size, origin: tuple[float, float] | None = None):
	"""
	changes the scale of an image's txt file label

	:param image_file: path to image, yolov8 format
	:type image_file: .txt file
	:param out_file: path to output scaled image
	:param scale_amount: amount to scale image
	:type scale_amount: float
	:param image_size: the width and height of the image in pixels, (w, h)
	:param origin: the origin point that the image scales from: (x, y), none = center
	:type origin: tuple ( float, float )
	"""
	if isinstance(image_size, tuple) and len(image_size) == 2:
		h, w = image_size
	elif isinstance(image_size, str) or isinstance(image_size, Path):
		img = cv2.imread(str(image_size))
		h, w = img.shape[:2]
	else:
		print("Wrong parameter for image_size, use path or (height, width)")
		return None
	
	ox, oy = origin if origin is not None else (w / 2.0, h / 2.0)

	# Normalise origin to [0,1] space for easier arithmetic
	ox_n = ox / w
	oy_n = oy / h

	input_lines = Path(label_file).read_text().strip().splitlines()
	output_lines = []

	for line in input_lines:
		if not line.strip():
			continue
		parts = line.split()
		if len(parts) != 5:
			continue
		class_id = parts[0]
		cx, cy, bw, bh = map(float, parts[1:5])

		# Apply the same affine shift as the image (all in normalised coords)
		cx_new = scale_amount * (cx - ox_n) + ox_n
		cy_new = scale_amount * (cy - oy_n) + oy_n
		bw_new = scale_amount * bw
		bh_new = scale_amount * bh

		# Clamp edges and shrink box accordingly
		x1 = cx_new - bw_new / 2
		x2 = cx_new + bw_new / 2
		y1 = cy_new - bh_new / 2
		y2 = cy_new + bh_new / 2

		x1c, x2c = max(0.0, x1), min(1.0, x2)
		y1c, y2c = max(0.0, y1), min(1.0, y2)

		cx_f = (x1c + x2c) / 2
		cy_f = (y1c + y2c) / 2
		bw_f = x2c - x1c
		bh_f = y2c - y1c

		if bw_f <= 0 or bh_f <= 0:
			continue
		
		# calculating area for spec 1.2.1
		old_area = bw_new * bh_new
		new_area = bw_f * bh_f
		if new_area < old_area * AREA_RATIO_MIN:
			continue
		# calculating pixels for spec 1.2.2 & 1.2.3
		pw = bw_f * w
		ph = bh_f * h
		if pw < PX_MIN_W or ph < PX_MIN_H:
			continue
		if pw > PX_MAX_W or ph > PX_MAX_H:
			continue
		
		output_lines.append(
			f"{class_id} {cx_f:.6f} {cy_f:.6f} {bw_f:.6f} {bh_f:.6f}"
		)

	Path(out_file).write_text("\n".join(output_lines) + "\n" if output_lines else "")
	return out_file