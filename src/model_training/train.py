#	@file: train.py
#	@brief: trains the YOLO model in Tempest

import os
import sys
import shutil
from ultralytics import YOLO
from roboflow_datasets import robo_arg_parse, roboflow_download

if __name__ == "__main__":
	model = YOLO("yolov8m.pt")
    
	config = robo_arg_parse()
	if config is None:
		print("Failed to configure arguments")
		sys.exit(1)

	dataset = roboflow_download(config)

	if not os.path.exists(config.get("dataset")):
		print(f"No dataset at dir: {config.get("dataset")}")
		sys.exit(1)
	if "save" not in config:
		print(f"No save directory for model")
		sys.exit(1)

	os.makedirs(config.get("save"), exist_ok = True)

	dataset_dir = os.path.join(config.get("directory"), config.get("dataset"))
	results = model.train(data = dataset_dir,
                          epochs = 10,
                          imgsz = 640,
                          patience = 10,
                          cache = False,
                          seed = 17,
                          device = [0,1,2,3],
                          project = config.get("save"),
                          )