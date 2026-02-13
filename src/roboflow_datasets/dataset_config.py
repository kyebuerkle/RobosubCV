#	@file: dataset_config.py
#	@brief: this houses the Config class for the entire RobosubCV project

import roboflow
import shutil
import os
import json
from dataset_module import roboflow_login

class Config:
	#	Roboflow config settings
	path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "roboflow_config.json")
	workspace  = ""
	project = ""
	version = 1
	dataset_dir = os.path.abspath(f"" \
							   f"{os.path.dirname(os.path.abspath(__file__))}" \
							   f"/../../data")
	api_key = None

	"""--------Private------------"""
	def __init__(self, args = None):
		"""
		Constructor method
		
		:params args: dict of arguments to configure
		"""
		if args:
			if "path" in args:
				self.path = args.get("path")
			elif "config_path" in args:
				self.path = args.get("config_path")
			
			self.__load_file()
			self.config_dict(args)
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
		return os.path.abspath(os.path.join(self.dataset_dir, f"{self.workspace}-{self.project}-v{self.version}"))
		
	def to_dict(self):
		"""
		Converts current Config to a dictionary		
		"""
		#	TODO: check if any keys are the same
		ret = {
				"directory" : self.dataset_dir,
				"dataset" : self.get_dataset(),
				"workspace" : self.workspace,
				"project" : self.project,
				"version" : self.version,
				"key" : self.api_key
			}		
		return ret
	
	def save_file(self):
		"""
		Saves the current Config to .json file
		"""
		if not os.path.exists(self.path):
			return False
		
		save_dict = self.to_dict()
		save_dict.pop("key")
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
				self.workspace = val
			elif (key == "project"):
				self.project = val
			elif (key == "version"):
				self.version = val
			elif (key == "url"):
				self._parse_url(val)
			elif (key == "api_key" or key == "key"):
				self.api_key = val
			elif (key == "directory"):
				self.dataset_dir = val

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