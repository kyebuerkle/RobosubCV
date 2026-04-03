#	@file: util_module.py
#	@brief: general utility functions

import argparse

def parse_float_list(value):
	"""Parse comma-separated list of floats, return None on single value of 0"""
	if value == 0 or value.lower() == "none":
		return None
	try:
		return [float(x.strip()) for x in value.split(',')]
	except ValueError:
		raise argparse.ArgumentTypeError(f"Invalid float list: {value}")
	
def parse_int_list(value):
	"""Parse comma-separated list of floats, return None on single value of 0"""
	if value == 0 or value.lower() == "none":
		return None
	try:
		return [int(x.strip()) for x in value.split(',')]
	except ValueError:
		raise argparse.ArgumentTypeError(f"Invalid int list: {value}")