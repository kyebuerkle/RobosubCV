#	upload_dataset.py
#	Kye Buerkle
#	brief: uploads the augmented dataset back to roboflow
#	usage: upload_dataset.py [options]
#	NOTE: this process takes forever... 3,000 images takes 20-30 min
#	NOTE: this process OVERWRITES the project in roboflow. make sure to adjust at least the version

import sys
from dataset_module import robo_arg_parse
from dataset_config import Config

if __name__ == "__main__":
	arg_dict= robo_arg_parse()
	config = Config(arg_dict)
	
	config.roboflow_upload(config)