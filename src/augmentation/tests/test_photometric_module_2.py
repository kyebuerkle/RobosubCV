#	@file: test_photometric_module_2.py
#	@brief: Spec tests that relate to the blur, contrast, and hue_shift augmentations
#	If you use the --save argument, it will save all the images in this test

import pytest
import cv2
import numpy as np
from pathlib import Path
from augmentation.photometric_module_2 import gaussian_blur, motion_blur, contrast, hue_shift


def get_mean_percent_difference(expected, actual):
	"""gets the mean percent diff of np arrays"""

	diff = np.abs(actual.astype(np.float32) - expected.astype(np.float32))
	percent_diff = diff / 255
	mean_percent_diff = percent_diff.mean()

	return float(mean_percent_diff)

def blur_of_image(image_path):
	"""returns the image as a float32 array for blur comparisons"""
	image_cv = cv2.imread(str(image_path))
	return image_cv.astype(np.float32)

def contrast_of_image(image_path):
	"""returns the image as float32 for contrast comparisons"""
	image_cv = cv2.imread(str(image_path))
	return image_cv.astype(np.float32)

def hue_of_image(image_path):
	"""returns the hue channel (H) from the HSV image"""
	image_cv = cv2.imread(str(image_path))
	image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)
	return image_hsv[:, :, 0]		#	H channel only

def assert_blur_val(image1, image2, amount):
	"""
	Assert that the output image is plausibly blurred relative to the input.
	Uses Laplacian variance as a sharpness proxy — a blurred image should
	have lower variance than the original (for amount < 1, edge case for amount >= 1).

	:return sharpness_ratio, shape_bool: ratio of output/input sharpness, shape match bool
	"""
	img1 = cv2.imread(str(image1))
	img2 = cv2.imread(str(image2))

	shape_bool = (img1.shape == img2.shape)

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

	sharpness1 = cv2.Laplacian(gray1, cv2.CV_32F).var()
	sharpness2 = cv2.Laplacian(gray2, cv2.CV_32F).var()

	#	avoid divide-by-zero for completely flat test images
	if sharpness1 == 0:
		sharpness_ratio = 1.0
	else:
		sharpness_ratio = sharpness2 / sharpness1

	return sharpness_ratio, shape_bool

def assert_contrast_val(image1, image2, amount):
	"""
	assert statements that compare image1 with image2 and asserts the expected contrast.

	Contrast is applied as: out = mean + (pixel - mean) * amount
	percent error is the minimal required error allowed (this is caused from rounding)
	:return percent_error, bool: calculated percent error, bool if shape passed
	"""
	img1 = cv2.imread(str(image1)).astype(np.float32)
	img2 = cv2.imread(str(image2)).astype(np.float32)

	shape_bool = (img1.shape == img2.shape)

	mean = img1.mean(axis=(0, 1), keepdims=True)
	expected = np.clip(mean + (img1 - mean) * amount, 0, 255).astype(np.uint8)
	actual = np.clip(img2, 0, 255).astype(np.uint8)

	calc_percent_error = get_mean_percent_difference(expected, actual)

	return calc_percent_error, shape_bool

def assert_hue_val(image1, image2, amount):
	"""
	assert statements that compare image1 with image2 and asserts the expected hue shift.

	Hue shift is: (amount - 1.0) * 180 degrees in OpenCV HSV (0-179 range), wrapped.
	percent error is the minimal required error allowed (this is caused from rounding)
	:return percent_error, bool: calculated percent error, bool if shape passed
	"""
	img1 = cv2.imread(str(image1))
	img2 = cv2.imread(str(image2))

	shape_bool = (img1.shape == img2.shape)

	hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype(np.int32)
	hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.int32)

	shift = int(round((amount - 1.0) * 180.0))
	expected_hue = ((hsv1[:, :, 0] + shift) % 180).astype(np.uint8)
	actual_hue   = hsv2[:, :, 0].astype(np.uint8)

	#	compare only the hue channel, normalised over 180 (not 255)
	diff = np.abs(actual_hue.astype(np.float32) - expected_hue.astype(np.float32))
	calc_percent_error = float((diff / 180).mean())

	return calc_percent_error, shape_bool


# ─────────────────────────────────────────────────────
#  Gaussian Blur
# ─────────────────────────────────────────────────────

PERCENT_ERROR_MIN = 0.05

@pytest.mark.parametrize("blur_amount", [0.5, 1.0, 1.5, 2.0, 0.1])
def test_gaussian_blur_various_amounts(tmp_image, output_dir, blur_amount, error_csv_writer):
	"""Test gaussian_blur with different amounts"""

	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{blur_amount}_gblur.png"

	out_str = gaussian_blur(tmp_image, output_image, blur_amount)

	sharpness_ratio, shape = assert_blur_val(tmp_image, output_image, blur_amount)
	error_csv_writer(output_image.name, "gaussian_blur", blur_amount, 1.0 - sharpness_ratio)

	debug_blur(tmp_image, output_image, blur_amount, "gaussian_blur")

	assert str(out_str) == str(output_image)
	assert shape
	#	for amount=1 we still apply sigma=10, so output is blurrier than input in all cases
	#	sharpness ratio must be < 1 (output less sharp than input), except on flat images
	if blur_amount > 0:
		assert sharpness_ratio <= 1.05		#	allow tiny floating-point margin

@pytest.mark.parametrize("blur_amount", [0.0])
def test_gaussian_blur_zero_amount(tmp_image, output_dir, blur_amount, error_csv_writer):
	"""amount=0 means sigma=0, image should be returned unchanged"""

	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{blur_amount}_gblur.png"

	out_str = gaussian_blur(tmp_image, output_image, blur_amount)

	img1 = cv2.imread(str(tmp_image))
	img2 = cv2.imread(str(output_image))
	percent_error = get_mean_percent_difference(img1, img2)
	error_csv_writer(output_image.name, "gaussian_blur", blur_amount, percent_error)

	assert str(out_str) == str(output_image)
	assert percent_error <= PERCENT_ERROR_MIN


# ─────────────────────────────────────────────────────
#  Motion Blur
# ─────────────────────────────────────────────────────

@pytest.mark.parametrize("blur_amount", [0.5, 1.0, 1.5, 2.0, 0.1])
def test_motion_blur_various_amounts(tmp_image, output_dir, blur_amount, error_csv_writer):
	"""Test motion_blur (horizontal, default angle=0) with different amounts"""

	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{blur_amount}_mblur.png"

	out_str = motion_blur(tmp_image, output_image, blur_amount)

	sharpness_ratio, shape = assert_blur_val(tmp_image, output_image, blur_amount)
	error_csv_writer(output_image.name, "motion_blur", blur_amount, 1.0 - sharpness_ratio)

	debug_blur(tmp_image, output_image, blur_amount, "motion_blur")

	assert str(out_str) == str(output_image)
	assert shape
	if blur_amount > 0:
		assert sharpness_ratio <= 1.05

@pytest.mark.parametrize("angle", [0, 45, 90, 135])
def test_motion_blur_various_angles(tmp_image, output_dir, angle, error_csv_writer):
	"""Test motion_blur at different angles with a fixed moderate amount"""

	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{angle}deg_mblur.png"

	out_str = motion_blur(tmp_image, output_image, 1.0, angle=angle)

	sharpness_ratio, shape = assert_blur_val(tmp_image, output_image, 1.0)
	error_csv_writer(output_image.name, "motion_blur_angle", angle, 1.0 - sharpness_ratio)

	assert str(out_str) == str(output_image)
	assert shape
	assert sharpness_ratio <= 1.05

@pytest.mark.parametrize("blur_amount", [0.0])
def test_motion_blur_zero_amount(tmp_image, output_dir, blur_amount, error_csv_writer):
	"""amount=0 means length=0, image should be returned unchanged"""

	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{blur_amount}_mblur.png"

	out_str = motion_blur(tmp_image, output_image, blur_amount)

	img1 = cv2.imread(str(tmp_image))
	img2 = cv2.imread(str(output_image))
	percent_error = get_mean_percent_difference(img1, img2)
	error_csv_writer(output_image.name, "motion_blur", blur_amount, percent_error)

	assert str(out_str) == str(output_image)
	assert percent_error <= PERCENT_ERROR_MIN


# ─────────────────────────────────────────────────────
#  Contrast
# ─────────────────────────────────────────────────────

@pytest.mark.parametrize("con_amount", [0.7, 1.0, 1.3, 2.0, 0.0, 0.5])
def test_contrast_various_amounts(tmp_image, output_dir, con_amount, error_csv_writer):
	"""Test contrast function with different amounts"""

	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{con_amount}_con.png"

	out_str = contrast(tmp_image, output_image, con_amount)

	percent_error, shape = assert_contrast_val(tmp_image, output_image, con_amount)
	error_csv_writer(output_image.name, "contrast", con_amount, percent_error)

	debug_contrast(tmp_image, output_image, con_amount)

	assert str(out_str) == str(output_image)
	assert shape
	assert percent_error <= PERCENT_ERROR_MIN


# ─────────────────────────────────────────────────────
#  Hue Shift
# ─────────────────────────────────────────────────────

@pytest.mark.parametrize("hue_amount", [0.5, 1.0, 1.5, 0.0, 2.0])
def test_hue_shift_various_amounts(tmp_image, output_dir, hue_amount, error_csv_writer):
	"""Test hue_shift with different amounts"""

	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{hue_amount}_hue.png"

	out_str = hue_shift(tmp_image, output_image, hue_amount)

	percent_error, shape = assert_hue_val(tmp_image, output_image, hue_amount)
	error_csv_writer(output_image.name, "hue_shift", hue_amount, percent_error)

	debug_hue(tmp_image, output_image, hue_amount)

	assert str(out_str) == str(output_image)
	assert shape
	assert percent_error <= 0.10


# ─────────────────────────────────────────────────────
#  Debug helpers
# ─────────────────────────────────────────────────────

def debug_blur(image1, image2, amount, func_name="blur"):
	"""Print a breakdown of sharpness before/after blur."""
	img1 = cv2.imread(str(image1))
	img2 = cv2.imread(str(image2))

	gray1 = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY).astype(np.float32)
	gray2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY).astype(np.float32)

	sharpness1 = cv2.Laplacian(gray1, cv2.CV_32F).var()
	sharpness2 = cv2.Laplacian(gray2, cv2.CV_32F).var()

	print(f"\n--- {func_name} debug (amount={amount}) ---")
	print(f"  input  sharpness (Laplacian var): {sharpness1:.2f}")
	print(f"  output sharpness (Laplacian var): {sharpness2:.2f}")
	print(f"  ratio output/input: {sharpness2/sharpness1:.4f}" if sharpness1 > 0 else "  input is flat (sharpness=0)")

def debug_contrast(image1, image2, amount):
	"""Print a breakdown of what the contrast check is actually seeing."""
	img1 = cv2.imread(str(image1)).astype(np.float32)
	img2 = cv2.imread(str(image2)).astype(np.float32)

	mean = img1.mean(axis=(0, 1), keepdims=True)
	expected = np.clip(mean + (img1 - mean) * amount, 0, 255)

	print(f"\n--- contrast debug (amount={amount}) ---")
	print(f"  input  pixel: min={img1.min():.1f} max={img1.max():.1f} mean={img1.mean():.1f}")
	print(f"  output pixel: min={img2.min():.1f} max={img2.max():.1f} mean={img2.mean():.1f}")
	print(f"  expected:     min={expected.min():.1f} max={expected.max():.1f} mean={expected.mean():.1f}")

	diff = np.abs(img2 - expected)
	print(f"  abs diff: mean={diff.mean():.2f} max={diff.max():.2f}")
	print(f"  % error:  mean={diff.mean()/255*100:.2f}% max={diff.max()/255*100:.2f}%")

def debug_hue(image1, image2, amount):
	"""Print a breakdown of what the hue check is actually seeing."""
	img1 = cv2.imread(str(image1))
	img2 = cv2.imread(str(image2))

	hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV).astype(np.int32)
	hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV).astype(np.int32)

	shift = int(round((amount - 1.0) * 180.0))
	expected_hue = (hsv1[:, :, 0] + shift) % 180
	actual_hue = hsv2[:, :, 0]

	diff = np.abs(actual_hue - expected_hue)

	print(f"\n--- hue_shift debug (amount={amount}, shift={shift}°) ---")
	print(f"  input  H: min={hsv1[:,:,0].min()} max={hsv1[:,:,0].max()} mean={hsv1[:,:,0].mean():.1f}")
	print(f"  output H: min={actual_hue.min()} max={actual_hue.max()} mean={actual_hue.mean():.1f}")
	print(f"  expected: min={expected_hue.min()} max={expected_hue.max()} mean={expected_hue.mean():.1f}")
	print(f"  abs diff: mean={diff.mean():.2f} max={diff.max():.2f}")
	print(f"  % error (over 180): mean={diff.mean()/180*100:.2f}% max={diff.max()/180*100:.2f}%")