#	@file: configure_settings.py
#	@brief: script that you can use to configure your Config file
#	TODO: I want this to bring you through the steps, logging into roboflow, settings
#		augmentation values, selecting output location, ext...

from pathlib import Path
from roboflow_datasets import Config, robo_arg_parse

def main():
	arg_dict = robo_arg_parse()

	config_file = Path(Config.path)
	if config_file.exists():
		if arg_dict.get("yes", False):
			yes = "y"
		else:
			yes = input("Do you with to delete your old config file? [y/n] ")
		if yes.lower() == "y":
			config_file.unlink()
	
	config_obj = Config(arg_dict)
	#	TODO: add a print function in Config for debugging
	config_obj.save_file()

if __name__ == "__main__":
	main()