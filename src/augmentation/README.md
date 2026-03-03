#		Augmentation

Image Augmentaion Subsystem. This system augments the training and validation images to emulate environmental changes during the competition. There are two main processes: Geometric augmentation, Photometric augmentation. Geometric augments the scale, and crop of the images. Photometric augments the exposure and saturation of the images.

##		Usage

###		Photometric 

- `change_exposure(image_file, out_file, amount)` : function that augments a single image
  - image_file : string or Path to image you want to augment
  - out_file : name of output
  - amount : float value how much, 0.5 -> 50%, 1 -> the same 100%, 1.5 50% more or 150%
- `change_saturation(image_file, out_file, amount)` : function that augments a single image
  - image_file : string or Path to image you want to augment
  - out_file : name of output
  - amount : float value how much, 0.5 -> 50%, 1 -> the same 100%, 1.5 50% more or 150%
- `dir_change_exposure(input, output, list)` : function that does augmentations through an entire directory
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
- `dir_change_saturation(input, output, list)` : function that does augmentations through an entire directory
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
- `yolo_change_exposure(input, output, list)` : function that does augmentations through an entire yolov8 formatted dataset
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
- `yolo_change_saturation(input, output, list)` : function that does augmentations through an entire yolov8 formatted dataset
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images

##		Tests

`poetry run pytest` to run through unit tests

use `poetry run pytest --help` to show all the arguments
use `poetry run pytest --save` to save all the output images in an output directory
use `poetry run pytest --dataset path/to/dataset` to run the tests on a specific dataset