#	@file: photometric_module.py
#	@brief: This script holds the common photometric augmentation functions
#		Eg. change_exposure & change_saturation

import cv2
import numpy as np

def change_exposure(image_file, out_file, exposure_amount: float):
	"""
	changes the exposure of an image and saves it as a seperate file
	
	:param image_file: input image file
	:param exposure_amount: % to change the exposure, >2 is clamped to 2
	:type exposure_amount: float
	:param out_file: save it under this file name
	:type out_file: string
	"""
	image = cv2.imread(str(image_file))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_file}")
	
	# Special case: no change
	if exposure_amount == 1.0:
		cv2.imwrite(str(out_file), image)
		return out_file
	
	unclamped = image.astype(np.float32)
	multiplied_img_float = unclamped * exposure_amount
	multiplied_img = np.clip(multiplied_img_float, 0, 255).astype(np.uint8)
	cv2.imwrite(str(out_file), multiplied_img)
	return out_file

	"""
	#	Another way to write the exposure function
	# Special case: no change
	if exposure_amount == 1.0:
		image = cv2.imread(str(image_file))
		cv2.imwrite(str(out_file), image)
		return out_file

	image = cv2.imread(str(image_file))

	hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
	hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 2] * exposure_amount, 0, 255) 	# Modify only exposure / value channel
	hsv_img = hsv_img.astype(np.uint8)
	final_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

	cv2.imwrite(str(out_file), final_image)
	return out_file
	"""
	
def change_saturation(image_file, out_file, saturation_amount: float):
	"""
	changes the saturation of an image and saves it as a seperate file

	:param image_file: input image file
	:param out_file: save it under this file name
	:param saturation_amount: % change the saturation, >2 is clamped 
	:type saturation_amount: float
	"""		

	image = cv2.imread(str(image_file))
	if image is None:
		raise FileNotFoundError(f"Could not read image: {image_file}")
	
	# Special case: no change
	if saturation_amount == 1.0:
		cv2.imwrite(str(out_file), image)
		return out_file

	hsv_img = cv2.cvtColor(image, cv2.COLOR_BGR2HSV).astype(np.float32)
	hsv_img[:, :, 1] = np.clip(hsv_img[:, :, 1] * saturation_amount, 0, 255) 	# Modify only saturation channel
	hsv_img = hsv_img.astype(np.uint8)
	final_image = cv2.cvtColor(hsv_img, cv2.COLOR_HSV2BGR)

	cv2.imwrite(str(out_file), final_image)
	return out_file