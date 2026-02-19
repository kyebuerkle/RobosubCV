import pytest
import shutil
from pathlib import Path

def pytest_addoption(parser):
	parser.addoption(
		"--save",
		action="store_true",
		default=False,
		help="Save test output images to permanent location"
	)
	parser.addoption(
        "--dataset",
        action="store",
        default=None,
        help="Root directory of a dataset"
    )

def pytest_sessionstart(session):
	if session.config.getoption("--save"):
		save_dir = Path(__file__).parent / "output"
		if save_dir.exists():
			print("Cleaning output file for tests")
			shutil.rmtree(save_dir)

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
