#		dataset_config

This script holds the `Config` class, used in the model_training scripts to save settings and get them for training a model. 

##		Config

I won't write everything, just what is important.

- Members: path, workspace, project, version, api_key, save_dir, exposure, saturation, resize
- get_dataset() : returns the full dataset path
- get_yaml() : returns the *data.yaml* file in the dataset
- to_dict() : returns the Config class members as a dictionary
- config_dict(dict) : takes, 'dict' dictionary input and sets members to it's values
- save_file() : will save the `configuration.json` file to `RobosubCV/src/`
- run_augmentations(**kwargs) : runs the yolo_dataset augmentations based on the members of the object AND the **kwargs input
- roboflow_download(format, yes) : downloads the dataset based on members, format -> coco or yolov8, yes = False will give a prompt if a dataset needs to be deleted to replace
- roboflow_upload() : uploads the dataset to roboflow