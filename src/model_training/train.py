#	@file: train.py
#	@brief: trains the YOLO model in Tempest

import os
import sys
import shutil
from ultralytics import YOLO
from roboflow_datasets import robo_arg_parse, Config

if __name__ == "__main__":
	model = YOLO("yolov8m.pt")
    
	config = Config(robo_arg_parse())
	if config is None:
		print("Failed to configure arguments")
		sys.exit(1)

	#	format should be yolov8
	dataset = config.roboflow_download(format = "yolov8", yes = True)
	dataset_dir = config.get_dataset()
	yaml_file = os.path.join(dataset_dir, "data.yaml")

	if not os.path.exists(yaml_file):
		print(f"No dataset at dir: {dataset_dir}")
		sys.exit(1)
	if not config.save_dir or config.save_dir == "":
		print(f"No save directory for model")
		sys.exit(1)
	os.makedirs(config.save_dir, exist_ok = True)

	results = model.train(data = yaml_file,
                          epochs = 10,
                          imgsz = 640,
                          patience = 10,
                          cache = False,
                          seed = 17,
                          device = [0,1],
                          project = config.save_dir,
                          )