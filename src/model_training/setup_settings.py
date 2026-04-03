#	@file: setup_settings.py
#	@brief: This script will be used to setup the users config file, and roboflow login

import argparse
import sys
import roboflow
import json
from pathlib import Path

from general_lib import parse_float_list, parse_int_list
from roboflow_datasets import Config, roboflow_login

def _parse_args() -> dict:
	"""
	Parses the arguments. Returns dictionary of the arguments (use it to update to not re-write)

	Arguments:
	-exp/--exposure		comma seperated exposure list
	-sat/--saturation	comma seperated saturation list
	-res/--resize		comma seperated resize list
	-o/--output			output results directory
	-d/--dataset		dataset save directory
	"""
	parser = argparse.ArgumentParser(
		description="Set up the configuration file for model training",
		epilog="Use options to only change one portion of the configuration"
		)
	
	parser.add_argument(
		'-exp', "--exposure",
		metavar="LIST",
		type = parse_float_list,
		help = "comma seperated list of exposure augmentations"
		)
	parser.add_argument(
		'-sat', "--saturation",
		metavar="LIST",
		type = parse_float_list,
		help = "comma seperated list of saturation augmentations"
		)
	parser.add_argument(
		'-res', "--resize",
		metavar="LIST",
		type = parse_float_list,
		help = "comma seperated list of resize augmentations"
		)
	parser.add_argument(
		'-o', "--output",
		metavar="DIR",
		type = str,
		help = "Output directory of the YOLO model results"
		)
	parser.add_argument(
		'-d', "--dataset",
		metavar="DIR",
		type = str,
		help = "directory to save the datasets to"
		)
	parser.add_argument(
		'-p', "--print",
		action="store_true",
		help = "prints the configuration.json file"
		)
	parser.add_argument(
		"--epochs", 
		metavar="INT",
		type=int, 
		help="epoch paramter for model training"
		)
	parser.add_argument(
		"--patience", 
		metavar="INT", 
		type=int,
		help="patience paramter for model training"
		)
	parser.add_argument(
		"--device", 
		metavar="LIST", 
		type=parse_int_list, 
		help="device paramter for model creation"
		)
	parser.add_argument(
		"--model",
		metavar="YOLO MODEL",
		type=str,
		help="model paramter for model training, yolov8m.pt <- medium model, replace the m with n for nano, s for small, l for large, and x for extra large"
		)
	
	args = parser.parse_args()
	ret = {}
	
	if (args.exposure):
		ret["exposure"] = args.exposure
	if (args.saturation):
		ret["saturation"] = args.saturation
	if (args.resize):
		ret["resize"] = args.resize
	if (args.output):
		ret["save"] = Path(args.output).resolve()
	if (args.dataset):
		ret["directory"] = Path(args.dataset).resolve()
	if (args.print):
		ret["print"] = args.print
	if (args.epochs):
		ret["epochs"] = args.epochs
	if (args.patience):
		ret["patience"] = args.patience
	if (args.device):
		ret["device"] = args.device
	if (args.model):
		ret["model"] = args.model
	
	if not ret:
		return None
	else:
		return ret

def _check_yes(answer: str) -> bool:
	"""checks if an input is y or yes"""
	ans = answer.strip().lower()
	if ans == "y" or ans == "yes":
		return True
	else:
		return False

def alturnate_main(arg_dict):
	"""
	alternate_main is used to update the existing settings, main will overwrite everything
	"""
	printing = arg_dict.pop("print", False)
	if printing:
		print("Updating configuration.json with these arguments:")
		print(json.dumps(arg_dict, indent=4))

	config = Config()
	config.config_dict(arg_dict)
	config.save_file()

	print("Saved configuration")
	if printing:
		print(json.dumps(config.to_dict(), indent=4))

def main():
	"""
	Main script
	
	This script steps through seting up the config file and roboflow login
	1. log into roboflow (choice of api or link)
	2. Model results directory (optional)
	3. Dataset save directory (optional)
	4. Augmentation types (optional)
		Saturation
		Exposure
		Resize
	"""
	robosubcv_dir = Path(__file__).parent.parent.parent.absolute()

	print("")
	print("Thank you for using the RobosubCV repository to train your YOLOv8 model!")
	print("Follow the next steps to setup your configuration.json file")
	print("If you don't want to save this information, use arguments in the 'train.sbatch' script in the 'sbatch' folder")
	print("(to figure out the arguments run: 'poetry run python src/train.py -h', arguments always overwrite the save file)")
	print("----------------------------------------------------------------------------------------------------------------")
	print("")

	#	Roboflow login
	print("First you need to log into Roboflow in this terminal, you can either use an API key or login with Roboflow's url prompt.")
	answer = input("Do you wish to use an API key (press enter or n for Roboflow prompt)? [y/n] ")

	api_key = None
	if _check_yes(answer):
		print("To find your API key go to app.roboflow.com and go into your 'account settings' (bottom left corner with profile pic)")
		print("Under 'Workspace' in the top left, select which project has the datasets you want to train")
		print("Then go to the 'API Keys' section and copy your 'Private Key'. NOTE: the configuration.json file that this will save to is not secure, use ^C to exit and start over")
		api_key = input("Paste your API Key here: ")
		rf = roboflow_login(api_key=api_key)
	else:
		rf = roboflow_login()
	
	if rf is None:
		print("Log in failed, try again or use API")
		sys.exit(1)
	else:
		print("Successful Login!")

	try:
		workspace = rf.workspace()
		print(f"Default workspace: {workspace.name}")
	except:
		print(f"Couldn't find a default workspace with loggin")
	
	#	Save Results
	result_str = input("\nType in your prefered results directory for the YOLO output to go to (press Enter for default): ")
	result_path = Path(result_str)
	if not result_str.strip():
		result_path = robosubcv_dir / "results/"
	elif not result_path.exists():
		answer = input(f"Directory: {result_path}, does not exist. Do you want it to be created? [y/n] ")
		if _check_yes(answer):
			print(f"Creating, and saving directory {result_path}...")
			result_path.mkdir(parents=True)
		else:
			print(f"Not creating, saving default directory. Re-run script to change again.")
			result_path = robosubcv_dir / "results/"
	
	result_path = result_path.absolute().resolve()
	if not result_path.exists():
		print(f"ERROR: failed to save dataset directory at: {result_path}, exiting...")
		sys.exit(1)
	print(f"Saving models to {result_path}\n")

	#	Save Directory
	dir_str = input("Type in your prefered save directory for image training datasets (press Enter for default): ")
	dir_path = Path(dir_str)
	if not dir_str.strip():
		dir_path = robosubcv_dir / "data/"
	elif not dir_path.exists():
		answer = input(f"Directory: {dir_path}, does not exist. Do you want it to be created? [y/n] ")
		if _check_yes(answer):
			print(f"Creating, and saving directory {dir_path}...")
			dir_path.mkdir(parents=True)
		else:
			print(f"Not creating, saving default directory. Re-run script to change again.")
			dir_path = robosubcv_dir / "data/"
	
	dir_path = dir_path.absolute().resolve()
	if not dir_path.exists():
		print(f"ERROR: failed to save dataset directory at: {dir_path}, exiting...")
		sys.exit(1)
	print(f"Saving datasts to {dir_path}\n")

	#	Augmentation
	print("Finally, enter the amount of augmentation you want for each of the three types")
	print("Press 'Enter' for default augmentation, Press '0' for no augmentation")
	print("For each type make a list of comma seperated values (no spaces) of the % you want it augmented")
	print("Example: 'Saturation: 0.5,1,1.5,2' will augment the dataset 4 times at 50%, 100% (no change to the image), 150%, and 200%")

	values = input("\n    Saturation: ")
	if not values or values.strip() == "" or values == None:
		saturation = [0.7, 1.0, 1.3]
		print(f"Saving defaults: {saturation}, -30%, original, +30%")
	elif values.strip() == "0":
		saturation = None
		print(f"No Saturation augmentation, equivilant to 1.0")
	else:
		saturation = parse_float_list(values)
		print(f"Saving values: {saturation}")

	values = input("\n    Exposure: ")
	if not values or values.strip() == "" or values == None:
		exposure = [0.615, 1.0, 1.385]
		print(f"Saving defaults: {exposure}, -37.5%, original, +38.5%. (This is ±3 EV for RealSense camera)")
	elif values.strip() == "0":
		exposure = None
		print(f"No Exposure augmentation, equivilant to 1.0")
	else:
		exposure = parse_float_list(values)
		print(f"Saving values: {exposure}")

	values = input("\n    Resize: ")
	if not values or values.strip() == "" or values == None:
		resize = None
		print(f"Saving defaults: {resize}, no default resize augmentation")
	elif values.strip() == "0" or values.strip == "1" or values.strip == "1.0":
		resize = None
		print(f"No resize augmentation, equivilant to 1.0")
	else:
		resize = parse_float_list(values)
		print(f"Saving values: {resize}")

	#	saving file
	print(f"\nSaving config file at {Config.path}")
	config = Config(
		api_key = api_key,
		save_dir = result_path,
		dataset_dir = dir_path,
		saturation = saturation,
		exposure = exposure,
		resize = resize
		)
	config.save_file()
	print("Finished setup, final file values: ")
	print(json.dumps(config.to_dict(), indent=4))

if __name__ == "__main__":
	arg_dict = _parse_args()
	if arg_dict:
		alturnate_main(arg_dict)
	else:
		main()