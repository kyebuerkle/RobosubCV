#	@file: run_augmentation_dataset.py
#	@brief: script to run dataset augmentation functions

from pathlib import Path

from general_lib import loading, Animations
import augmentation.config as config
from augmentation import yolo_augment
from augmentation.augment_strategy import MODE_ALL, MODE_RANDOM, MODE_CALC
from augmentation._shared_module import get_args

def main(args):
	"""Main script"""
	input_path  = Path(args.input_dir).resolve()
	output_path = Path(args.output_dir).resolve()

	#	Build aug_config from whatever flags were passed — skip empty lists
	aug_config = {}
	if args.exposure:      aug_config['exposure']      = args.exposure
	if args.contrast:      aug_config['contrast']      = args.contrast
	if args.resize:        aug_config['resize']        = args.resize
	if args.motion_blur:   aug_config['motion_blur']   = args.motion_blur
	if args.gaussian_blur: aug_config['gaussian_blur'] = args.gaussian_blur
	if args.hue_shift:     aug_config['hue_shift']     = args.hue_shift
	if args.saturation:    aug_config['saturation']    = args.saturation

	if not aug_config:
		print("No augmentation arguments provided — nothing to do.")
		return

	if config.VERBOSE:
		print(f"Augmentation mode:   {args.augment_mode}  (num={args.augment_num})")
		print(f"Active aug types:    {list(aug_config.keys())}")
		for k, v in aug_config.items():
			print(f"  {k}: {v}")

	#	determine if dataset is coco or yolov8
	yaml_path = input_path / "data.yaml"
	coco_path = input_path / "test/_annotations.coco.json"
	if yaml_path.exists():
		if config.VERBOSE:
			print("Dataset is yolov8 formatting")
		yolo_augment(
			input_dataset  = str(input_path),
			output_dataset = str(output_path),
			aug_config     = aug_config,
			mode           = args.augment_mode,
			num            = args.augment_num,
		)
	elif coco_path.exists():
		if config.VERBOSE:
			print("Dataset is coco formatting")
		print("Nothing to do for coco format yet")
	else:
		print(f"No data.yaml or coco annotation found in {input_path}")
		return

	if config.VERBOSE:
		print("Finished augmenting dataset!")

if __name__ == "__main__":
	args = get_args()
	if config.VERBOSE:
		main(args)
	else:
		with loading("Augmenting ", Animations.coen_fight):
			main(args)