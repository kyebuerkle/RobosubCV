#	@file: test_photometric_module.py
#	@brief: Spec tests that relate to the photometric augmentations
#	If you use the --save argument, it will save all the images in this test

import pytest
import cv2
import numpy as np
from pathlib import Path
from augmentation.photometric_module import change_saturation, change_exposure

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}

def pytest_generate_tests(metafunc):
	"""
	If a test uses `tmp_image`, expand any directory entries in the fixture's
	params into individual image file paths at collection time.
	"""
	if "tmp_image" not in metafunc.fixturenames:
		return

	raw_params = ["random", "middle"]  # keep your base params here

	# Add your asset directory — swap this path for your real one
	asset_dir_str = (
        metafunc.config.getoption("--asset-dir", default=None)
        or metafunc.config.getini("asset_dir")
    	)
	asset_dir = Path(asset_dir_str).resolve()
	if asset_dir.is_dir():
		image_files = sorted(
			p for p in asset_dir.iterdir()
			if p.suffix.lower() in IMAGE_EXTENSIONS
		)
		raw_params.extend(image_files)  # each file becomes its own param

	metafunc.parametrize("tmp_image", raw_params, indirect=True)

@pytest.fixture
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
	elif request.param == "middle":
		image_file = tmp_path / "tmp_mid_image.png"
		image = np.full(shape = (100, 100, 3), fill_value = 127)
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

def exposure_of_image(image):
	"""returns the list of exposure / brightness values"""
	image_cv = cv2.imread(str(image))
	image_hsv = cv2.cvtColor(image_cv, cv2.COLOR_BGR2HSV)

	return image_hsv[:, :, 2]

def assert_saturation_val(image1, image2, expected_saturation, percent_error = 0.05):
	"""
	assert statements that compare image1 with image2 and asserts the expecetd saturation

	percent error is the minimal required error allowed (this is caused from rounding)
	:return percent_error: calculated percent error
	"""
	normal_sat = saturation_of_image(image1)
	compare_sat = saturation_of_image(image2)

	#	Images must be the same size
	assert normal_sat.shape == compare_sat.shape

	expected_sat = np.clip(normal_sat * expected_saturation, 0, 255)
	calc_percent_error = get_mean_percent_error(expected_sat, compare_sat)
	assert calc_percent_error <= percent_error

	return calc_percent_error

def assert_exposure_val(image1, image2, expected_exposure, percent_error = 0.05):
	"""
	assert statements that compare image1 with image2 and asserts the expecetd exposure

	percent error is the minimal required error allowed (this is caused from rounding)
	:return percent_error: calculated percent error
	"""
	normal_exp = exposure_of_image(image1)
	compare_exp = exposure_of_image(image2)

	#	Images must be the same size
	assert normal_exp.shape == compare_exp.shape

	expected_exp = np.clip(normal_exp * expected_exposure, 0, 255)
	calc_percent_error = get_mean_percent_error(expected_exp, compare_exp)
	assert calc_percent_error <= percent_error

	return calc_percent_error

@pytest.mark.parametrize("sat_amount", [0.7, 1, 1.3, 2, 0, -0.5])
def test_saturation_various_amounts(tmp_image, output_dir, sat_amount, error_csv_writer):
	"""Test saturation function with different amounts"""
	
	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{sat_amount}_sat.png"

	out_str = change_saturation(tmp_image, output_image, sat_amount)
	assert str(out_str) == str(output_image)

	percent_error = assert_saturation_val(tmp_image, output_image, sat_amount, 0.05)
	error_csv_writer(output_image.name, "saturation", sat_amount, percent_error)


@pytest.mark.parametrize("exp_amount", [0.615, 1, 1.385, 2, 0, -0.5])
def test_exposure_various_amounts(tmp_image, output_dir, exp_amount, error_csv_writer):
	"""Test exposure function with different amounts"""
	
	image_name = Path(tmp_image)
	output_image = output_dir / f"{image_name.stem}_{exp_amount}_exp.png"

	out_str = change_exposure(tmp_image, output_image, exp_amount)
	assert str(out_str) == str(output_image)

	percent_error = assert_exposure_val(tmp_image, output_image, exp_amount, 0.05)
	error_csv_writer(output_image.name, "exposure", exp_amount, percent_error)
