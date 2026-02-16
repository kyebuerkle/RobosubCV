#	download_dataset.py
#	Kye Buerkle
#	brief: downloads a roboflow dataset
#	usage: download_dataset.py [options]

import argparse
import sys
import os
from dataset_module import CONFIG_PATH, config_dict, roboflow_download 

# --------- Main Function ----------
if __name__ == "__main__":
	# parse the arguments
	parser = argparse.ArgumentParser(
		prog = "download_dataset.py",
		description = "downloads a dataset from roboflow",
		argument_default = argparse.SUPPRESS
		)

	parser.add_argument('-u', "--url", help = "Input the URL of the roboflow project")
	parser.add_argument('-d', "--directory", help = "output directory")
	parser.add_argument('-w', "--workspace", help = "roboflow workspace")
	parser.add_argument('-p', "--project", help = "Project ID from rboflow")
	parser.add_argument('-v', "--version", help = "versions of project, default 1", type=int)
	parser.add_argument('-k', "--key", help = "api key for Roboflow login")
	parser.add_argument('-f', "--format", help = "model format of images")
	parser.add_argument('-y', "--yes", help = "accepts the overwrite without waiting for user input", action="store_true")
	# TODO: parser.add_argument('-c', "--configuration", help = "configuration file", default = CONFIG_PATH)
	args = parser.parse_args()
	arg_dict = vars(args)

	config = config_dict(CONFIG_PATH, **arg_dict)
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
