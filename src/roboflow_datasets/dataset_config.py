#	@file: dataset_config.py
#	@brief: this houses the Config class for the entire RobosubCV project

import roboflow
import shutil
import os
import json

from roboflow_datasets import roboflow_login
from augmentation import yolo_change_exposure, yolo_change_saturation, yolo_change_resize

class Config:
	#	Roboflow config settings
	path = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "../configuration.json"))
	workspace  = ""
	project = ""
	version = 1
	dataset_dir = os.path.abspath(f"" \
							   f"{os.path.dirname(os.path.abspath(__file__))}" \
							   f"/../../data")
	api_key = None
	save_dir = ""

	#	Augmentation settings
	saturation = []
	exposure = []
	resize = []

	#	model creation parameters
	patience = 15
	epochs = 40
	device = [0, 1]
	model = "yolov8m.pt"

	"""--------Private------------"""
	def __init__(self, args = None, **kwargs):
		"""
		Constructor method
		
		:params args: dict of arguments to configure
		:params kwarg: argument to configure (same as dict)
		"""
		params = {}
		params.update(kwargs)
		if args is not None:
			params.update(args)

		if params:
			if "path" in params:
				self.path = params.get("path")
			elif "config_path" in params:
				self.path = params.get("config_path")
			
			self.__load_file()
			self.config_dict(params)
			self.save_file()
		else:
			self.__load_file()

	def __load_file(self):
		"""
		Loads settings from the .json file at self.path

		NOTE: this overwrites the current Config settings
		"""
		if not os.path.exists(self.path):
			return False
		with open(self.path, "r") as f:
			json_file = json.load(f)
			self.config_dict(json_file)

		return True
	
	"""---------Protected---------"""
	def _parse_url(self, url):
		"""
		Parses the url into the workspace, project, and version

		:params url: srt of URL
		"""
		temp_space = url.split("/")
		try:
			self.version = int(temp_space[-1])
		except:
			return False
		self.workspace = temp_space[-3]
		self.project = temp_space[-2]
		
		return True

	#	Public -------------

	def get_dataset(self):
		"""
		Returns the full dataset path

		:returns str: The path to the dataset
		"""
		if not self.project or self.project == "":
			return ""
		return os.path.abspath(os.path.join(self.dataset_dir, f"{self.project}-v{self.version}"))
	
	def get_yaml(self):
		"""
		Returns the data.yaml file for training
		"""
		yaml = os.path.join(self.get_dataset(), "data.yaml")
		if not os.path.exists(yaml):
			return None
		
		return yaml
	
	def get_save_dir(self, new_save = None):
		"""
		Returns the save directory, None on fail
		"""
		if new_save:
			self.save_dir = new_save

		if not self.save_dir or self.save_dir == "":
			return None
		os.makedirs(self.save_dir, exist_ok = True)

		return self.save_dir
		
	def to_dict(self):
		"""
		Converts current Config to a dictionary		
		"""
		ret = {
				"directory" : self.dataset_dir,
				"dataset" : self.get_dataset(),
				"workspace" : self.workspace,
				"project" : self.project,
				"version" : self.version,
				"save" : self.save_dir,
				"key" : self.api_key,
				"saturation" : self.saturation,
				"exposure" : self.exposure,
				"resize" : self.resize,
				"patience" : self.patience,
				"epochs" : self.epochs,
				"device" : self.device,
				"model" : self.model
			}	
		return ret
	
	def save_file(self):
		"""
		Saves the current Config to .json file
		"""
		save_dict = self.to_dict()
		#save_dict.pop("key")
		with open(self.path, "w") as f:
			json.dump(save_dict, f, indent = 2)
		
		return True
	
	def config_dict(self, dict):
		"""
		Configures the Config from a dictionary
		
		:params dict: Dictionary
		:returns True: when successful
		:returns False: when certain config (like workspace and project) don't exist
		"""
		for key, val in dict.items():
			#if (key == "path" or key == "config_path"):
			#	self.path = val
			if (val is None or val == ""):
				pass
			elif (key == "workspace"):
				self.workspace = str(val)
			elif (key == "project"):
				self.project = str(val)
			elif (key == "version"):
				self.version = val
			elif (key == "url"):
				self._parse_url(val)
			elif (key == "api_key" or key == "key"):
				self.api_key = val
			elif (key == "directory"):
				self.dataset_dir = str(val)
			elif (key == "save_dir" or key == "save_path" or key == "save"):
				self.save_dir = str(val)
			elif (key == "saturation"):
				self.saturation = val
			elif (key == "exposure"):
				self.exposure = val
			elif (key == "resize"):
				self.resize = val
			elif (key == "epochs"):
				self.epochs = val
			elif (key == "patience"):
				self.patience = val
			elif (key == "device"):
				self.device = val
			elif (key == "model"):
				self.model = val

		if (self.workspace is None) or (self.workspace == ""):
			return False
		if (self.project is None) or (self.project == ""):
			return False
		if (self.version < 0) or (not isinstance(self.version, int)):
			return False
		if (not os.path.exists(self.dataset_dir)):
			return False
		if (not os.path.exists(self.path)):
			return False
		if (self.save_dir == "" or self.save_dir is None):
			return False

		return True
	
	def roboflow_download(self, format = "coco", yes = False):
		"""
		downloads roboflow dataset
		
		:param format: dataset format [coco | yolov8] 
		:param yes: bool, true to delete dataset to replace, false to ask for user input
		:returns dataset: the roboflow dataset object
		"""
		if os.path.exists(self.get_dataset()):
			print("Dataset already exists, Do you wish to overwrite?")
			if yes:
				response = "y"
			else:
				response = input("[y/n] ")
			if response.lower() == "y":
				try:
					shutil.rmtree(self.get_dataset())
				except Exception as e:
					print(f"Failed to delete previous dataset at: {self.get_dataset()}\n{e}")
					return False
		
		#	login to roboflow & download dataset
		try:
			rf = roboflow_login(api_key = self.api_key)
			project = rf.workspace(self.workspace).project(self.project)
			version = project.version(self.version)
			dataset = version.download(model_format = format, location = self.get_dataset())
		except Exception as e:
			print(f"Failed to download dataset from config file: {self.get_dataset()}\n{e}")
			return False

		self.save_file()
		return dataset
	
	def roboflow_upload(self):
		"""
		Uploads dataset to roboflow
		"""
		if not os.path.exists(self.get_dataset()):
			print(f"No path exists to upload: {self.get_dataset()}")
			return
		
		try:
			rf = roboflow_login(api_key = self.api_key)
			workspace = rf.workspace(self.workspace)
			workspace.upload_dataset(
				self.get_dataset(),
				self.project(),
				#dataset_format = config.get("format"),
				project_license = "MIT",
				project_type = "object-detection"
				)
		except Exception as e:
			print(f"Failed to upload dateset from config file: {self.path}")
			return
		
	def run_augmentations(self, **kwargs):
		"""
		Runs the augmentations in the Config, note adds 1 to the version for new dataset

		:return dataset: dataset path to train, if no augmentations then it stays as get_dataset()
		"""
		if "saturation" in kwargs:
			self.saturation = kwargs.get("saturation")
		if "exposure" in kwargs:
			self.exposure = kwargs.get("exposure")
		if "resize" in kwargs:
			self.resize = kwargs.get("resize")

		input_dataset = self.get_dataset()
		self.version += 1
		augmented_dataset = self.get_dataset()

		if isinstance(self.saturation, list) and self.saturation:
			yolo_change_saturation(
				input_dataset, augmented_dataset,
				self.saturation, "{file}_sat{ind}{ext}"
				)
			input_dataset = augmented_dataset
		if isinstance(self.exposure, list) and self.exposure:
			yolo_change_exposure(
				input_dataset, augmented_dataset,
				self.exposure, "{file}_exp{ind}{ext}"
				)
			input_dataset = augmented_dataset
		if isinstance(self.resize, list) and self.resize:
			yolo_change_resize(
				input_dataset, augmented_dataset,
				self.resize, "{file}_res{ind}{ext}"
				)
		
		return input_dataset