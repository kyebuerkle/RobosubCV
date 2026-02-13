#	download_dataset.py
#	Kye Buerkle
#	brief: downloads a roboflow dataset
#	usage: download_dataset.py [options]

import sys
import os
from dataset_config import Config
from dataset_module import robo_arg_parse

# --------- Main Function ----------
if __name__ == "__main__":
	# parse the arguments
	arg_dict = robo_arg_parse()
	config = Config(arg_dict)
	
	dataset = config.roboflow_download(format = arg_dict.get("format", "coco"), yes = arg_dict.get("yes", False))
	if dataset is None:
		print("Failed to download dataset")
		sys.exit(1)
	
	print(f"Path to dataset: {dataset.location}")
	print(f"Dataset name: {dataset.name}")
	print(f"Dataset version: {dataset.version}")
