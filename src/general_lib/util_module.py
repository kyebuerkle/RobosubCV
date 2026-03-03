#	@file: util_module.py
#	@brief: general utility functions

import argparse

def parse_float_list(value):
	"""Parse comma-separated list of floats"""
	try:
		return [float(x.strip()) for x in value.split(',')]
	except ValueError:
		raise argparse.ArgumentTypeError(f"Invalid float list: {value}")