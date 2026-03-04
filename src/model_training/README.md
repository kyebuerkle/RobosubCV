#		model_training

this holds the source code that trains the YOLO model

##		train.py

This script actually trains the YOLO model.
It utilizes the `roboflow_datasets` lib in src to download the roboflow data, and train the yolo model

These are the general steps its walks through:
1. Uses arguments to generate a Config object to configure all settings
2. Downloads the dataset specified in the config settings (yolov8 format)
3. Runs augmentations on the downloaded dataset
   1. Saves augmentations to new dataset +1 version (saves to config, but not to file)
4. Trains YOLO model on the augmented dataset

**MAKE SURE** this script is run in an environment with CUDA, torchvision and ultralytics. These are how the YOLO model knows what hardware to utilize on Tempest, or any computer, to train efficiently.

###		Usage

train.py works the same way as the download_dataset.py
```
poetry install
poetry run python train.py [options]
```

*[options]* is the same as download_dataset. but the important ones are:
- `-k` : the API key to log into Roboflow
- `-u` : the URL of the Roboflow dataset you want to train the model on
- `-exp` : exposure values, comma seperated Eg: `-exp 1.5,1,-0.5`
- `-sat` : saturation values, comma seperate Eg: `-sat 1.5,1,-0.5`

##    setup_settings.py

This script takes a user through a couple steps to setup their `configureation.json` file. 
Run:
```
poetry run python setup_settings.py
```

**MAKE SURE** to run this script before schedualing your `train.sbatch` job, unless you used arguments in that script. Otherwise it won't work.