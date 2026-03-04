#		general_lib

This is the general utilities library for the RobosubCV project. The main thing about this lib is that it doesn't require imports from the other packages in RobosubCV. 

There are 2 modules:
1. `util_module.py` is general utility
2. `pytest_util_module.py` is pytest utility

Plus a script: `draw_labels.py` that draws the labels in a directory onto images from that same directory.

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

###		draw_labels

This script takes in a directory of images and labels, and saves the images with the lables drawn on them into an output directory. 

Run:
```
poetry run python draw_labels.py input_dir output_dir [options]
```
Use `-h` as an option to figure out how to use

Options:
- input_dir : input directory with your images and your labels
- output_dor : output directory to save to
- -c, --color : the RGB color value of the label boxs
  
Ex:
```
poetry run python draw_labels.py ../../assets/test-images ../../data/test-labels -c 255 20 150
```

##		Tests

To run general unit tests, run:
```
poetry run pytest
```

This will run through all the tests in the tests dir of the module you are in
run this in the main directory (RobosubCV) to run every test accross src