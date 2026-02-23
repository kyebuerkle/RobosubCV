#	@file: geometric_module.py
#	@brief: this script holds geometric augmentation functions
#		Eg. change_scale

import cv2
import numpy as np

def change_scale(image_file, out_file, scale_amount: float):
	"""
	changes the scale of an image

	:param image_file: path to image
	:param out_file: path to output scaled image
	:param scale_amount: amount to scale image
	:type scale_amount: float
	"""
	pass

def yolo_scale_label(label_file, out_file, scale_amount: float):
	"""
	changes the scale of an image's txt file label

	:param image_file: path to image, yolov8 format
	:type image_file: .txt file
	:param out_file: path to output scaled image
	:param scale_amount: amount to scale image
	:type scale_amount: float
	"""
	pass