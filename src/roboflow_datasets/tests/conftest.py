from pathlib import Path
import pytest


def pytest_addoption(parser):
    parser.addoption(
        "--dataset",
        action="store",
        default=None,
        help="Root directory of the COCO dataset"
    )


@pytest.fixture(scope="session")
def dataset_root(request):
    root = request.config.getoption("--dataset")

    if root is None:
        #pytest.fail(
        #    "Missing --dataset argument\n"
        #    "Example: pytest --dataset data/my-dataset"
        #)
        pytest.skip("Missing --dataset, skipping test...")
        return None

    path = Path(root).resolve()

    if not path.exists():
        pytest.fail(f"Dataset root does not exist: {path}")

    return path
