#	download_dataset.py
#	Kye Buerkle
#	brief: downloads a roboflow dataset
#	usage: download_dataset.py [options]

import sys
import os
from dataset_module import roboflow_download, robo_arg_parse

# --------- Main Function ----------
if __name__ == "__main__":
	# parse the arguments
	config = robo_arg_parse()
	if config is None:
		print("Failed to configure arguments")
		sys.exit(1)
	
	dataset = roboflow_download(config)
	if dataset is None:
		print("Failed to download dataset")
		sys.exit(1)
	
	print(f"Path to dataset: {dataset.location}")
	print(f"Dataset name: {dataset.name}")
	print(f"Dataset version: {dataset.version}")
