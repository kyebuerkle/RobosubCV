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
- `dir_change_exposure(input, output, list, name_format)` : function that does augmentations through an entire directory
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
  - name_format : format to name the augmented files
- `dir_change_saturation(input, output, list, name_format)` : function that does augmentations through an entire directory
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
  - name_format : format to name the augmented files
- `yolo_change_exposure(input, output, list, name_format)` : function that does augmentations through an entire yolov8 formatted dataset
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
  - name_format : format to name the augmented files
- `yolo_change_saturation(input, output, list, name_format)` : function that does augmentations through an entire yolov8 formatted dataset
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
  - name_format : format to name the augmented files

###   Geometric

- `change_scale(image_file, out_file, amount, origin)` : function that augments a single image
  - image_file : string or Path to image you want to augment
  - out_file : name of output
  - amount : float value how much, 0.5 -> 50%, 1 -> the same 100%, 1.5 50% more or 150%
  - origin : origin point in pixels of where the scale is applied on the image
- `yolo_scale_label(label_file, out_file, amount, image, origin)` : function that adjusts the label of an image (coincides with change_scale)
  - label_file : string or Path to label file
  - out_file : name of output
  - amount : float value how much, 0.5 -> 50%, 1 -> the same 100%, 1.5 50% more or 150%
  - image : original image file, or tuple (width, height) size of image
  - origin : origin point in pixels of where the scale is applied on the image
- `dir_change_scale(input, output, list, name_format, origin)` : function that does augmentations through an entire directory
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
  - name_format : format to name the augmented files
  - origin : origin point in pixels of where the scale is applied on the image
- `yolo_change_resize(input, output, list, name_format, origin)` : function that does augmentations through an entire yolov8 formatted dataset
  - input : input string or Path to the directory
  - output : output directory, if the same as input replaces it
  - list : list of values to augment all the images
  - name_format : format to name the augmented files
  - origin : origin point in pixels of where the scale is applied on the image

###   name_format

name_format is a string input parameter into these functions, used to save the files in a specific manner. How it works is like a format string: "string_val{variable}". The variables to pick from are:
- file : the original file name without the extenstion
  - name_format = "{file}" Ex. "test.png" -> "test"
- val : is the amount of augmentation to 3 decimal points
  - name_format = "{val}" Ex. augmented with 0.3498 -> "0.349"
- ind : is the index of the augmentation
  - name_format = "{ind}" Ex. augmentations [1.0, 0.54] -> "0" & "1"
- ext : is the file extention, normally .png
  - name_format = "{ext}" Ex. "test.png" -> ".png"

Full example:
there are 2 images: "test1.png" and "test1.2.png" being augmented with values: [0.5, 1, 2.3759].
name_format = "Augmented\_{file}\_with\_{val}_at-{ind}{ext}"

Output augmented images:
"Augmented\_test1\_with\_0.500\_at-0.png"
"Augmented\_test1\_with\_1.000\_at-1.png"
"Augmented\_test1\_with\_2.375\_at-2.png"
"Augmented\_test1.2\_with\_0.500\_at-0.png"
"Augmented\_test1.2\_with\_1.000\_at-1.png"
"Augmented\_test1.2\_with\_2.375\_at-2.png"

---

##		Tests

`poetry run pytest` to run through unit tests

use `poetry run pytest tests/test_module.py` to run only a specific test script with any options
use `poetry run pytest --help` to show all the arguments
use `poetry run pytest --save` to save all the output images in an output directory
use `poetry run pytest --dataset path/to/dataset` to run the tests on a specific dataset
use `poetry run pytest --csv path/to/csv_file.csv` to save spec test data to a csv file

###   Spec Tests 1.3.1 & 1.3.2

Run:
```
poetry run pytest tests/test_photometric_module.py --save --csv photometric_output.csv
```
This will test the saturation, and the exposure on all the images in the `assets/test-images/` directory. It will save all the augmentations to a `./tests/output/` folder, and save a csv file `./photometric_output.csv` with all the data to analyze for spec tests. During the test you could see green `.`s, yellow `s`s, or red `F`s. If any red `F` is in the test, then the spec test emidiatly fails. (to see more details for the test you can add `-v` to the command or `-vv` for even more details)
The augmentations it go through are:
- Saturation : [0.7, 1.0, 1.3, 0, 2, -0.5]
  - This is -30% to image, no change, +30% to image, no image, +100% or X2 to image, impossible augmentation
  - Spec 1.3.2: model must recognize objects with + and - 30% saturation
- Exposure : [0.615, 1.0, 1.385, 0, 2, -0.5]
  - For the RealSense camera (7.8 EV), this is equivilant to -3 EV, 0 EV, +3 EV, +7.8 EV, -7.8 EV, impossible augmentation
  - Spec 1.3.1: model must recognize object with + and - 3 EV

To perform the spec tests, open the `photometric_output.csv` file in Excel. 
The columns are as follows (there are no titles):
| File name | augmentation | amount | % error |
|-----------|--------------|--------|---------|

To pass Spec 1.3.1: 95% of all the exposure augmented images must have below 5% (0.05) error. There can only be 4% of images with between 5% and 15% error, to ensure accuracy. And 1% of images can be above 15% to pass.

To pass Spec 1.3.2: 95% of all the saturation augmented images must have below 5% (0.05) error. There can only be 4% of images with between 5% and 15% error, to ensure accuracy. And 1% of images can be above 15% to pass.

The fastest way to get this is to use `=MAX(D:D)` in an empty cell to the right of the data. If this maximum value falls below 0.05, then 100% of the images are below 5%, which passes these spec requirements. 
Typically all the exposure images are have 0% error, and all the saturation fall below 4%.

If the max is above 5% then we need to comb through the data itself. In Excel, highlight the D column with % errror values and select Home -> Conditional Formating -> Highlight Cells Rule -> Greater Than.... Plug in 0.05 to find all the cells above 5% error. Then calculate the percentages to see if it still fits the spec rules written above. 

###   Spec Test 1.2.1

Run:
```
poetry run pytest tests/test_geometric_module.py --save --csv geometric_output.csv
```
This will test the scale / resize augmentation on all the images with labels in the `assets/test-images/`. Any images without labels will be scipped, and show as a yellow `s` for that image in the test. It will save all the augmentations to a `./tests/output/` folder, and save a csv file `./geometric_output.csv` with all the data to analyze for spec tests. During the test you could see green `.`s, yellow `s`s, or red `F`s. If any red `F` is in the test, then the spec test emidiatly fails. (to see more details for the test you can add `-v` to the command or `-vv` for even more details)
The augmentations it goes through are:
- Resize / Scale : [0.5, 0.75, 1.0, 1.25, 1.5, 2.0]
  - zoom out 50%, zoom out 25%, normal, zoom in 25%, zoom in 50%, zoom in 100% or double
  - NOTE: negative values invert the image, so -0.5 is zooming out 50% but the image is also backwards and upside-down, these aren't tested because they are im-practical and not used for the script, or the spec
  - Spec 1.2.1: model must recognize 75% or more of object in frame
  - Spec 1.2.2: model must recognize minimum of 64x48
  - Spec 1.2.3: model must recognize maximum of 448x336
- The tests have set values
  - Ratio : 0.5 or 50%
  - Minimum : 5x5 pixels
  - Maximum : 1500x1500 pixels

To perform the spec test, open `./geometric_output.csv` in Excel.
The columns are as follows (there are no titles):
| File name | augmentation | amount | ratio lost | Did the label stay? | pixel width | pixel height |
|-----------|--------------|--------|------------|---------------------|-------------|--------------|

To pass Spec 1.2.1: 95% of the labels that have above the ratio requirement, below the maximum requirement, and above the minimum requirement, must stay in the image. Also, these labels must be vissibly accurate.

This spec has 2 requirements, and two analysis methods.
1. in the `./geometric_output.csv`, find out every label that has been Removed. A way to do this in excel is by selecting the E column (Did the label stay?) and pressing Home -> Conditional Formatting -> Highlight Cell Rule -> Text that Contains.... Fill in the box with 'Removed'. So now all the Removed is highlighted
   1. Every place that removed the label needs to have one or more of these requirements: ratio lost > 0.5, pixel width < 5 or > 1500, pixel height < 5 or > 1500
   2. Next, check that the Stayed labels are within the opposing requirements: ratio lost < 0.5, pixel width > 5 and < 1500, pixel height > 5 and < 1500
2. Go to the `./test/output/` folder, and look at each image and verify that the label is in the correct place. To know where the labels should be, check the image with "res_1.0_label" somewhere in the file name. This means it was resized by 1, (so no resizing) and the labels are placed on the image in bright green. Also, the current labels are around clearly defined objects and whatnot, so it should be clear if the labels are visually in the right places.

NOTE: there are other spec tests that this can do, like Spec 1.2.2 and 1.2.3. However, these are not official tests as of now.