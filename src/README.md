#		src

src directory (source) is where the source code lies. 

##		Direcotories

###		Setup

General setup of the subsystem directories
```
src/
├── subsystem/
│   ├── source_files.py
│   ├── README.md
│   └── tests/
│       └── pytest_files.py
```
Each subsytem is in it's own folder with their source files and tests directory. The tests directory holds the pytests for the source file (the unit tests). Add a readme to each directory to explain usage and descriptions. 

###		Current Dir Tree

```
RobosubCV/
├── assets/
│   └── test-images/
├── src/
│   ├── augmentation/
│   │   └── tests/
│   ├── general_lib/
│   │   └── tests/
│   ├── model_training/
│   │	└── tests/
│   └── roboflow_datasets/
│       └── tests/
└── sbatch/
```

##		augmentation

This directory holds the augmentation modules and the dataset scripts. These scripts are what augments the images, both photometricly and geometricaly. There are also dataset and directory scripts that will apply these augmentations over datasets and directories while keeping the desired format.

This directory also houses some spec tests: Spec 1.3.1, Spec 1.3.2, Spec 1.2.1. There is functionality for 1.2.2 and 1.2.3, however they are not official spec tests currently.

##		general_lib

This directory holds the general use functions. It has no inports / dependencies from other local packages in the src folder.

##		model_training

This directory holds the main model_training scripts: `train.py` and `setup_settings.py`. These scripts are directly associated with the `sbatch/train.sbatch` script. 
`train.py` actually trains the model with the augmentation parameters.
`setup_settings.py` is used to settup a user's configuration settings. These settings include: roboflow login, save directories, and augmentation parameters.

##		roboflow_datasets

This directory holds the scripts relating to roboflow, and dataset format. It also has the `Config` class used to save settings to files. 

##		confiugration.json

This directory is where your `configuration.json` will save to. This file holds the settings configured from `setup_settings.py`, run this script again to re-configure.
This saves data used in the `dataset_config.py` script that holds the `Config` class used to download, augment, and upload datasets for training the model.

- directory: where to save the dataset to
- dataset: name of the dataset
- workspace: roboflow workspace name id
- project: roboflow project name
- version: version of the dataset
- save: where to output the YOLO model results
- key: API key to log into roboflow. If **None** then it uses the roboflow website cache based on last login from terminal
- saturation: list of % saturation augmentation
- exposure: list of % exposure augmentation
- resize: list of % resize / scale augmentation