import pytest
import shutil
import csv
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