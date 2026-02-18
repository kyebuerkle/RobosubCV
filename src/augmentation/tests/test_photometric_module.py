#	@file: test_photometric_module.py
#	@brief: Spec tests that relate to the photometric augmentations

import pytest
import cv2
import numpy as np
from pathlib import Path
from augmentation.photometric_module import change_saturation, change_exposure



@pytest.fixture(params=["random", 
						Path(__file__).resolve().parent / "test_image.jpg"])
def tmp_image(request, tmp_path):
	"""
	create temperary image of random values, or output real image path

	:return image path: image path
	"""
	if request.param == "random":
		image_file = tmp_path / "tmp_rand_image.png"
		image = np.random.randint(0, 255, (100, 100, 3), dtype=np.uint8)
		cv2.imwrite(str(image_file), image)
		return image_file
	elif request.param == "one":
		image_file = tmp_path / "tmp_one_image.png"
		image = np.ones((100, 100, 3))
		cv2.imwrite(str(image_file), image)
		return image_file
	else:
		image_file = Path(request.param)
		if (not image_file.exists()):
			pytest.skip(f"Image not found: {str(image_file)}")
		return image_file
	
def get_mean_percent_error(expected, actual):
	"""gets the mean percent error of np arrays"""
	diff = np.abs(actual - expected)
	denominator = np.maximum(np.abs(expected), 0.001)  # Avoid division by zero
	percent_error = diff / denominator
	mean_percent_error = np.mean(percent_error)

	return float(mean_percent_error)

def saturation_of_image(image):
	"""returns the list of saturation values"""
	image_cv = cv2.imread(str(image))
	image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

	return image_hsv[:, :, 1]

def assert_saturation_val(image1, image2, expected_saturation, percent_error = 0.05):
	"""
	assert statements that compare image1 with image2 and asserts the expecetd saturation

	percent error is the minimal required error allowed (this is caused from rounding)
	"""
	normal_sat = saturation_of_image(image1)
	compare_sat = saturation_of_image(image2)

	#	Images must be the same size
	assert normal_sat.shape == compare_sat.shape

	expected_sat = np.clip(normal_sat * expected_saturation, 0, 255)
	assert get_mean_percent_error(expected_sat, compare_sat) <= percent_error

def assert_hv_val(image1, image2, percent_error = 0.05):
	"""
	asserts the Hue and Value values to be the same
	(I realized that saturation is only one part that's different)
	"""
	image1_cv = cv2.imread(str(image1))
	image1_hsv = cv2.cvtColor(image1_cv, cv2.COLOR_BGR2HSV)
	image2_cv = cv2.imread(str(image2))
	image2_hsv = cv2.cvtColor(image2_cv, cv2.COLOR_BGR2HSV)

	image1_h = image1_hsv[:, :, 0]
	image1_v = image1_hsv[:, :, 2]
	image2_h = image2_hsv[:, :, 0]
	image2_v = image2_hsv[:, :, 2]

	assert image1_hsv.shape == image2_hsv.shape
	assert get_mean_percent_error(image1_h, image2_h) <= percent_error
	assert get_mean_percent_error(image1_v, image2_v) <= percent_error

@pytest.mark.parametrize("sat_amount", [0.5, 1.0, 1.5, 0, -0.5])
def test_saturation_various_amounts(tmp_image, sat_amount):
	"""Test saturation function with different amounts"""
	
	output_image = tmp_image.parent / f"output_{str(sat_amount)}.png"

	out_str = change_saturation(tmp_image, output_image, sat_amount)
	assert str(out_str) == str(output_image)

	assert_saturation_val(tmp_image, output_image, sat_amount, 0.05)
	# make sure the rest of the image stays the same
	#assert_hv_val(tmp_image, output_image, 0.05)