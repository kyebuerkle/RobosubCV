#	@file: test_photometric_module.py
#	@brief: Spec tests that relate to the photometric augmentations
#	If you use the --save argument, it will save all the images in this test

import pytest
import cv2
import numpy as np
from pathlib import Path
from augmentation.photometric_module import change_saturation, change_exposure

def get_mean_percent_difference(expected, actual):
	"""gets the mean percent diff of np arrays"""

	diff = np.abs(actual - expected)
	percent_diff = diff / 255	
	mean_percent_diff = percent_diff.mean()

	return float(mean_percent_diff)

def saturation_of_image(image):
	"""returns the list of saturation values"""
	image_cv = cv2.imread(str(image))
	image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

	return image_hsv[:, :, 1]

def exposure_of_image(image):
	"""returns the list of exposure / brightness values"""
	image_cv = cv2.imread(str(image))
	image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

	return image_hsv[:, :, 2]

def assert_saturation_val(image1, image2, expected_saturation):
	"""
	assert statements that compare image1 with image2 and asserts the expecetd saturation

	percent error is the minimal required error allowed (this is caused from rounding)
	:return percent_error, bool: calculated percent error, bool if shape passed
	"""
	normal_sat = saturation_of_image(image1).astype(np.float32)
	compare_sat = saturation_of_image(image2).astype(np.float32)

	#	Images must be the same size
	shape_bool = True if normal_sat.shape == compare_sat.shape else False

	expected_sat = np.clip(normal_sat * expected_saturation, 0, 255).astype(np.uint8)
	calc_percent_error = get_mean_percent_difference(expected_sat, compare_sat)

	return calc_percent_error, shape_bool

def assert_exposure_val(image1, image2, expected_exposure):
	"""
	assert statements that compare image1 with image2 and asserts the expecetd exposure

	percent error is the minimal required error allowed (this is caused from rounding)
	:return percent_error, bool: calculated percent error, bool if shape passed
	"""
	normal_exp = exposure_of_image(image1).astype(np.float32)
	compare_exp = exposure_of_image(image2).astype(np.float32)

	#	Images must be the same size
	shape_bool = True if normal_exp.shape == compare_exp.shape else False

	expected_exp = np.clip(normal_exp * expected_exposure, 0, 255).astype(np.uint8)
	calc_percent_error = get_mean_percent_difference(expected_exp, compare_exp)

	return calc_percent_error, shape_bool

PERCENT_ERROR_MIN = 0.05
@pytest.mark.parametrize("sat_amount", [0.7, 1, 1.3, 2, 0, -0.5])
def test_saturation_various_amounts(tmp_image, output_dir, sat_amount, error_csv_writer):
	"""Test saturation function with different amounts"""
	
	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{sat_amount}_sat.png"

	out_str = change_saturation(tmp_image, output_image, sat_amount)

	percent_error, shape = assert_saturation_val(tmp_image, output_image, sat_amount)
	error_csv_writer(output_image.name, "saturation", sat_amount, percent_error)

	debug_saturation(tmp_image, output_image, sat_amount)

	assert str(out_str) == str(output_image)
	assert shape
	assert percent_error <= PERCENT_ERROR_MIN

@pytest.mark.parametrize("exp_amount", [0.615, 1, 1.385, 2, 0, -0.5])
def test_exposure_various_amounts(tmp_image, output_dir, exp_amount, error_csv_writer):
	"""Test exposure function with different amounts"""
	
	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{exp_amount}_exp.png"

	out_str = change_exposure(tmp_image, output_image, exp_amount)

	percent_error, shape = assert_exposure_val(tmp_image, output_image, exp_amount)
	error_csv_writer(output_image.name, "exposure", exp_amount, percent_error)

	assert str(out_str) == str(output_image)
	assert shape
	assert percent_error <= PERCENT_ERROR_MIN

#	debugging
def debug_saturation(image1, image2, expected_saturation):
	"""Print a breakdown of what the saturation check is actually seeing."""
	normal_sat = saturation_of_image(image1)
	compare_sat = saturation_of_image(image2)
	expected_sat = np.clip(normal_sat * expected_saturation, 0, 255)

	print(f"\n--- saturation debug (scale={expected_saturation}) ---")
	print(f"  input  S: min={normal_sat.min():.1f} max={normal_sat.max():.1f} mean={normal_sat.mean():.1f}")
	print(f"  output S: min={compare_sat.min():.1f} max={compare_sat.max():.1f} mean={compare_sat.mean():.1f}")
	print(f"  expected: min={expected_sat.min():.1f} max={expected_sat.max():.1f} mean={expected_sat.mean():.1f}")

	diff = np.abs(compare_sat - expected_sat)
	print(f"  abs diff on valid pixels: mean={diff.mean():.2f} max={diff.max():.2f}")
	per_pixel_err = diff / 255
	print(f"  % error on valid pixels:  mean={per_pixel_err.mean()*100:.2f}% max={per_pixel_err.max()*100:.2f}%")