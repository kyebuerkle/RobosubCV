#	@file: photometric_module_2.py
#	@brief: This script holds additional photometric augmentation functions
#		Eg. gaussian_blur, motion_blur, contrast & hue_shift

import cv2
import numpy as np


def _odd(n: int) -> int:
	"""Ensure a kernel size is a positive odd integer (cv2 requirement)."""
	n = max(1, int(n))
	return n if n % 2 == 1 else n + 1


def gaussian_blur(image_file, out_file, amount: float):
	"""
	Blurs an image with a Gaussian kernel and saves it as a separate file.

	amount convention (consistent across all augmentation functions):
		1.0 = 100% = base blur (sigma = 10)
		0.5 = 50%  = sigma 5  (lighter blur)
		2.0 = 200% = sigma 20 (heavy blur)
		0.0 = no change (sigma = 0, image copied unchanged)

	:param image_file: input image file path
	:param out_file: save it under this file name
	:type out_file: string
	:param amount: % to scale the blur, base sigma = 10
	:type amount: float
	"""
	image = cv2.imread(str(image_file))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_file}")

	#	Special case: no blur
	if amount <= 0:
		cv2.imwrite(str(out_file), image)
		return out_file

	sigma = 10.0 * amount
	ksize = _odd(int(sigma * 6))		#	kernel covers ±3σ on each side
	blurred = cv2.GaussianBlur(image, (ksize, ksize), sigmaX=sigma, sigmaY=sigma)

	cv2.imwrite(str(out_file), blurred)
	return out_file


def motion_blur(image_file, out_file, amount: float, angle: float = 0.0):
	"""
	Simulates camera or subject motion with a directional streak blur
	and saves it as a separate file.

	amount convention:
		1.0 = 100% = 20 px streak length
		0.5 = 50%  = 10 px (subtle motion)
		2.0 = 200% = 40 px (heavy motion)
		0.0 = no change (length = 0, image copied unchanged)

	:param image_file: input image file path
	:param out_file: save it under this file name
	:type out_file: string
	:param amount: % to scale the streak length, base length = 20 px
	:type amount: float
	:param angle: direction of motion in degrees (0 = horizontal, 90 = vertical)
	:type angle: float
	"""
	image = cv2.imread(str(image_file))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_file}")

	length = int(round(20.0 * amount))

	#	Special case: no blur
	if length <= 0:
		cv2.imwrite(str(out_file), image)
		return out_file

	#	Build a 1-D horizontal motion kernel then rotate it to the desired angle
	kernel = np.zeros((length, length), dtype=np.float32)
	kernel[length // 2, :] = 1.0 / length

	cx, cy = length / 2.0, length / 2.0
	M = cv2.getRotationMatrix2D((cx, cy), angle, 1.0)
	kernel = cv2.warpAffine(kernel, M, (length, length))

	#	Re-normalise after rotation (interpolation redistributes weight)
	total = kernel.sum()
	if total > 0:
		kernel /= total

	blurred = cv2.filter2D(image, -1, kernel)

	cv2.imwrite(str(out_file), blurred)
	return out_file


def contrast(image_file, out_file, amount: float):
	"""
	Scales the contrast of an image around the per-channel mean
	and saves it as a separate file.

	Transform applied per pixel:  out = mean + (pixel - mean) * amount

	amount convention:
		1.0 = 100% = no change         (multiply by 1 -> identity)
		0.0 = 0%   = flat grey image   (all pixels collapse to the channel mean)
		0.5 = 50%  = half contrast     (washed out / low contrast)
		1.5 = 150% = boosted contrast  (punchy / high contrast)
		2.0 = 200% = very high contrast (clipping likely at extremes)

	:param image_file: input image file path
	:param out_file: save it under this file name
	:type out_file: string
	:param amount: % to scale the contrast, 1.0 = no change
	:type amount: float
	"""
	image = cv2.imread(str(image_file))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_file}")

	#	Special case: no change
	if amount == 1.0:
		cv2.imwrite(str(out_file), image)
		return out_file

	image_f32 = image.astype(np.float32)

	#	Per-channel mean so hue is not affected
	mean = image_f32.mean(axis=(0, 1), keepdims=True)		#	shape (1, 1, 3)

	scaled = mean + (image_f32 - mean) * amount
	scaled = np.clip(scaled, 0, 255).astype(np.uint8)

	cv2.imwrite(str(out_file), scaled)
	return out_file


def hue_shift(image_file, out_file, amount: float):
	"""
	Rotates the hue channel in HSV space and saves it as a separate file.

	OpenCV stores hue in [0, 180) (half the 360° wheel).
	Shift applied: hue_shift_degrees = (amount - 1.0) * 180

	amount convention:
		1.0 = 100% = 0° shift   -> no change
		0.0 = 0%   = -180° shift -> complementary / opposite colours
		1.5 = 150% = +90° shift  -> 90° clockwise on the colour wheel
		0.5 = 50%  = -90° shift  -> 90° counter-clockwise
		2.0 = 200% = +180° shift -> complementary colours (same result as 0.0)

	Saturation and value (brightness) are untouched.

	:param image_file: input image file path
	:param out_file: save it under this file name
	:type out_file: string
	:param amount: % to scale the hue rotation, 1.0 = no change
	:type amount: float
	"""
	image = cv2.imread(str(image_file))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_file}")

	#	Special case: no change
	if amount == 1.0:
		cv2.imwrite(str(out_file), image)
		return out_file

	image_f32 = image.astype(np.float32) / 255.0
	hsv_img = cv2.cvtColor(image_f32, cv2.COLOR_BGR2HSV)

	#	Shift in OpenCV hue units (0–179). Wrap around the wheel.
	shift = (amount - 1.0) * 180.0
	hsv_img[:, :, 0] = (hsv_img[:, :, 0] + shift) % 180.0

	final_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)
	final_image = np.clip(final_image * 255.0, 0, 255).astype(np.uint8)

	cv2.imwrite(str(out_file), final_image)
	return out_file