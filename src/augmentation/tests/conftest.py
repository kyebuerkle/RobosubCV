import pytest
import shutil
import csv
import cv2
import numpy as np
from pathlib import Path

def pytest_addoption(parser):
	parser.addoption(
		"--save",
		action="store_true",
		default=False,
		help="Save test output images to permanent location"
	)
	parser.addoption(
		"--csv",
		action="store",
		default=None,
		help="Path to save % error CSV results (e.g. results/errors.csv)",
	)
	try:
		parser.addoption(
			"--dataset",
			action="store",
			default=None,
			help="Root directory of a dataset"
		)
		parser.addoption(
			"--asset-dir",
			action="store",
			default=None,
			help="Path to directory of test images (overrides pytest.ini asset_dir)",
		)
		parser.addini(
			"asset_dir",
			help="Default path to directory of test images",
			default=None,
		)
	except ValueError:
		#	already passed
		pass

def pytest_sessionstart(session):
	if session.config.getoption("--save"):
		save_dir = Path(__file__).parent / "output"
		if save_dir.exists():
			print("Cleaning output file for tests")
			shutil.rmtree(save_dir)
	
@pytest.fixture(scope="session")
def error_csv_writer(request):
	"""
	Session-scoped fixture that opens a CSV file for writing arbitrary rows.
	Yields a callable: log_error(*args) — writes each arg as a column in order.
	Floats are formatted to 6 decimal places.
	If --error-csv is not passed, logging is a no-op.
	"""
	csv_path_str = request.config.getoption("--csv", default=None)

	if csv_path_str is None:
		yield lambda *args: None
		return

	csv_path = Path(csv_path_str)
	csv_path.parent.mkdir(parents=True, exist_ok=True)

	def format_value(v):
		if isinstance(v, float):
			return f"{v:.6f}"
		return v

	with open(csv_path, "w", newline="") as f:
		writer = csv.writer(f)

		def log_error(*args):
			writer.writerow([format_value(a) for a in args])
			f.flush()

		yield log_error

@pytest.fixture
def output_dir(request, tmp_path):
	"""
	Returns output directory based on --save flag
	- With --save: saves to tests/output/
	- Without --save: saves to tmp_path (auto-deleted)
	"""
	if request.config.getoption("--save"):
		# Save to permanent directory
		save_dir = Path(__file__).parent / "output"
		save_dir.mkdir(exist_ok=True)
		return save_dir
	else:
		# Save to temporary directory (auto-deleted)
		return tmp_path

@pytest.fixture(scope="session")
def dataset_root(request):
	root = request.config.getoption("--dataset")

	if root is None:
		return None

	path = Path(root).resolve()

	if not path.exists():
		pytest.fail(f"Dataset root does not exist: {path}")

	return path

#--- Generate temporary images ---

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
	