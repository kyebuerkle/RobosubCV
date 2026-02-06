#	upload_dataset.py
#	Kye Buerkle
#	brief: uploads the augmented dataset back to roboflow
#	usage: upload_dataset.py [options]
#	NOTE: this process takes forever... 3,000 images takes 20-30 min
#	NOTE: this process OVERWRITES the project in roboflow. make sure to adjust at least the version

import sys
from dataset_module import roboflow_upload, robo_arg_parse

if __name__ == "__main__":
	config = robo_arg_parse()
	if config is None:
		print("Failed to configure arguments")
		sys.exit(1)
	
	roboflow_upload(config)