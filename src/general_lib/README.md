#		general_lib

This is the general utilities library for the RobosubCV project. The main thing about this lib is that it doesn't require imports from the other packages in RobosubCV. 

There are 2 main modules:
1. `util_module.py` is general utility
2. `pytest_util_module.py` is pytest utility

##		Usage

###		General Utility

- `loading(text, animation)` : context that generates a loading animation in the terminal
  - text : string text that appears in front of animation
  - animation : List of 'frames' for animation
	```python
	with loading("Loading text ", [".", "..", "..."]):
		#	put code here
	```
- `Animations` : class that contains pre-set animations
  - `coen_fight` : my younger cousin, Coen, made this with me
	```python
	with loading("Coen-fight ", Animations.coen_fight):
		#	put code here
	```
- `parse_float_list(value)` : parses comma seperated string of floating point values
  - value : string of comma seperated float values
  - returns : list of values
	```python
	list = parse_float_point("0.5,1,1.5,-3.2")
	# list = [0.5, 1.0, 1.5, -3.2]
	```

###		Pytest Utility

- `validate_yolov8_dataset(dataset_root)` : function that validates a yolov8 formatted dataset (uses assert statements)
  - dataset_root : root path to the dataset that needs to be validated
	```python
	def test_this_path():
		path = "path/to/dataset"
		validate_yolov8_dataset(path)
	#	run pytest to validate
	```