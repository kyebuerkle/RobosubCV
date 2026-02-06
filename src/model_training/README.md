#		model_training

this holds the source code that trains the YOLO model

##		train.py

This script actually trains the YOLO model.
It utilizes the `roboflow_datasets` lib in src to download the roboflow data, and train the yolo model

**MAKE SURE** this script is run in an environment with CUDA, torchvision and ultralytics. These are how the YOLO model knows what hardware to utilize on Tempest, or any computer, to train efficiently.

###		Usage

train.py works the same way as the download_dataset.py
```
poetry run python train.py [options]
```

*[options]* is the same as download_dataset. but the important ones are:
- `-k` : the API key to log into Roboflow
- `-u` : the URL of the Roboflow dataset you want to train the model on
- `-y` : 'yes' automatically deletes the previous dataset with the same name (Tempest needs this activated)