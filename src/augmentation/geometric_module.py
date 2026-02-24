#	@file: geometric_module.py
#	@brief: this script holds geometric augmentation functions
#		Eg. change_scale

import cv2
import numpy as np

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

def yolo_scale_label(label_file, out_file, scale_amount: float):
	"""
	changes the scale of an image's txt file label

	:param image_file: path to image, yolov8 format
	:type image_file: .txt file
	:param out_file: path to output scaled image
	:param scale_amount: amount to scale image
	:type scale_amount: float
	"""
	pass