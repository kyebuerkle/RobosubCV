#		Augmentation

Image Augmentaion Subsystem. This system augments the training and validation images to emulate environmental changes during the competition. There are two main processes: Geometric augmentation, Photometric augmentation. Geometric augments the scale, and crop of the images. Photometric augments the exposure and saturation of the images.

##		Usage

...

##		Tests

run `poetry run pytest` for all the unit tests of geometric and photometric augmentations
use `--help` argument to see other options

`--save` saves images used in test