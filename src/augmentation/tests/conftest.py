import pytest
from pathlib import Path

def pytest_addoption(parser):
	parser.addoption(
		"--save",
		action="store_true",
		default=False,
		help="Save test output images to permanent location"
	)

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