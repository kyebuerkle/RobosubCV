#		roboflow

This directory has the scripts that upload and download images from roboflow.

##		Usage

###		Downloading datasets

run this command to download roboflow datasets

```
poetry run python download_dataset.py
```

This script will run through a couple steps to properly download the correct dataset.
use `-h` for help, `-w` for a workspace, `-p` for the project, `-v` for the version

- workspace: the name of the workspace with no capitals, and - instead of spaces
- project: project id, click on project in roboflow, press the 3 dots and copy "Project ID" 
- version: version of the dataset, use the most recent, otherwise this will download the first

###		Upload datasets

...

##		Tests

Roboflow upload and download won't be using 'unit tests', instead they will be tested through scripts since they require human input.

###		Download

1. Run the download_dataset.py script (go to Downloading datasets in Usage for steps)
2. copy the directory with the dataset (either from the command line or from file viewer)
3. run `poetry run pytest --dataset [paste directory] tests/test_coco_download.py ` to validate the format