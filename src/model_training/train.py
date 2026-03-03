#	@file: train.py
#	@brief: trains the YOLO model in Tempest

import sys
import json
from ultralytics import YOLO
from roboflow_datasets import robo_arg_parse, Config

def main(arg_dict, **kwargs):
	arg_dict.update(kwargs)
	model_file = arg_dict.get("model", None)
	if not model_file:
		print("No model selected to train")
		return False
	try:
		model = YOLO(model_file)
	except Exception as e:
		if e is FileNotFoundError or e is FileExistsError:
			print(f"Model file {model_file}, doesn't exist. Use 'yolov8m.pt' for online model")
			return False
	
	config = Config(arg_dict)
	if config is None:
		print("Failed to configure arguments")
		return False
	
	config.roboflow_download(format = "yolov8", yes = True)
	save_dir = config.get_save_dir()
	if not save_dir:
		print(f"No save directory for model")
		return False
	
	dataset_dir = config.run_augmentations()
	yaml_file = config.get_yaml()
	if not yaml_file:
		print(f"No yaml file found")
		return False
	
	print(f"\n"\
		f"────────────────────────────────────────────────────────────────────────────────────────────\n"\
		f"Training model on dataset yaml: {yaml_file}\n"\
		f"    Config settings:\n"
		f"{json.dumps(config.to_dict(), indent=4)}\n"
		f"────────────────────────────────────────────────────────────────────────────────────────────\n"\
		f"\n"
		)

	resutls = model.train(
		data 	= yaml_file,                
		epochs 	= arg_dict.get("epochs", 15),
		imgsz 	= arg_dict.get("imgsz", 640),
		patience = arg_dict.get("patience", 10),
		cache 	= False,
		seed 	= arg_dict.get("seed", 17),
		device 	= arg_dict.get("device", [0,1]),
		project = save_dir,
		)
	
	return True

if __name__ == "__main__":
	ret = main(robo_arg_parse(), model = "yolov8m.pt")
	if not ret:
		print("\n"\
			"!=======================!\n" \
			"! Failed to train model !\n" \
			"!=======================!\n",
			file = sys.stderr)
		sys.exit(1)
	else:
		print("Training success!")