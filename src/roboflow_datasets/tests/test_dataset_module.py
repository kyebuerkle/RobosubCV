#	@file: test_dataset_module.py
#	@brief: unit test for the dataset_module

import pytest
import os
import shutil
from roboflow_datasets import config_dict, save_json

CONFIG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "config_test.json")

class TestDatasetModule:

	@pytest.fixture
	def directory_in(self):
		dir_path = os.path.dirname(os.path.abspath(__file__))
		return os.path.abspath(f"{dir_path}/../../../data")

	#	tests if there is no config file
	def test_no_file(self, directory_in):
		input_args = {
			"workspace": "",
			"project": "",
			}
		results = config_dict(CONFIG_PATH, **input_args)
		assert results == None

		input_args = {
			"workspace": "test-workspace",
			"project": "",
			}
		results = config_dict(CONFIG_PATH, **input_args)
		assert results == None

		input_args = {
			"workspace": "",
			"project": "test-project",
			}
		results = config_dict(CONFIG_PATH, **input_args)
		assert results == None

		input_args = {
			"workspace": "test-workspace",
			"project": "test-project",
			}
		results = config_dict(CONFIG_PATH, **input_args)

		assert results == {
				"config_path": CONFIG_PATH,
				"directory" : directory_in,
				"dataset" : "test-project-v1",
				"workspace" : "test-workspace",
				"project" : "test-project",
				"version" : 1,
				"key" : None,
				"format" : "coco",
				"yes" : False,
				"verbose": False
			}
		
		#	saves the config file for future tests
		save_json(**results)

	def test_with_file(self, directory_in):
		if (not os.path.exists(CONFIG_PATH)):
			pytest.fail(f"Config file ({CONFIG_PATH}) doesn't exist")
			return
		
		input_args = {
			}
		results = config_dict(CONFIG_PATH, **input_args)
		assert results == {
				"config_path": CONFIG_PATH,
				"directory" : directory_in,
				"dataset" : "test-project-v1",
				"workspace" : "test-workspace",
				"project" : "test-project",
				"version" : 1,
				"key" : None,
				"format" : "coco",
				"yes" : False,
				"verbose": False
			}
		
	def test_cleanup(self):
		if os.path.exists(CONFIG_PATH):
			os.remove(CONFIG_PATH)

		assert not os.path.exists(CONFIG_PATH)