#	upload_dataset.py
#	Kye Buerkle
#	brief: uploads the augmented dataset back to roboflow
#	usage: upload_dataset.py [options]
#	NOTE: this process takes forever... 3,000 images takes 20-30 min
#	NOTE: this process OVERWRITES the project in roboflow. make sure to adjust at least the version

import sys
import argparse
from dataset_module import CONFIG_PATH, config_dict, roboflow_upload

if __name__ == "__main__":
	# parse the arguments
	parser = argparse.ArgumentParser(
		prog = "download_upload.py",
		description = "uploads a dataset to roboflow",
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
	
	roboflow_upload(config)