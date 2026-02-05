#		roboflow

This directory has the scripts that upload and download images from roboflow.

##		Usage

###		Downloading datasets

run this command to download roboflow datasets

```
poetry run python download_dataset.py -u [URL]
```

This script will run through a couple steps to properly download the correct dataset.
Where [URL] is the Roboflow project's URL. Go to [roboflow](app.roboflow.com), login, and locate the project you want to download, then just copy the URL into this command. Or use `-h` for help, `-w` for a workspace, `-p` for the project, `-v` for the version.

- workspace: the name of the workspace with no capitals, and - instead of spaces
- project: project id, click on project in roboflow, press the 3 dots and copy "Project ID" 
- version: version of the dataset, use the most recent, otherwise this will download the first

###		Upload datasets

run this command to upload roboflow datasets

```
poetry run python upload_dataset.py -u [URL]
```

Where [URL] is the url of your selected project. Copy and paste your project URL. 
Or instead use `-w` for the workspace, `-p` for the project and `-v` for the version.

> ![WARNING]
> this will overwrite your roboflow data, use with caution

##		Tests

Roboflow upload and download won't be using 'unit tests', instead they will be tested through scripts since they require human input.

###		Download

1. Run the download_dataset.py script (go to Downloading datasets in Usage for steps)
2. copy the directory with the dataset (either from the command line or from file viewer)
3. run `poetry run pytest --dataset [paste directory] tests/test_coco_download.py ` to validate the format

###		Upload

These steps assume you have the dataset installed alread (if not, run through Download)
1. Run `poetry run pytest --dataset [paste directory] tests/test_coco_download.py`
   1. or do `tests/test_yolo_download.py` if your format is for YOLOv8
2. Then upload your data using the upload steps above
3. Finally, check your (roboflow)[app.roboflow.com] to verify the upload

###		Unit Tests

For general unit tests run:
```
poetry run pytest
```

This will run through all the tests in the tests dir of the module you are in
run this in the main directory (RobosubCV) to run every test accross src