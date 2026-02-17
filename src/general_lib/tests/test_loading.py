#	@brief: tests the loading animation

import pytest
import argparse
import time

from general_lib.loading_animation import loading

def test_loading_animation(capsys):
	start = time.time()

	with loading():
		time.sleep(1)

	elapsed = time.time() - start
	captured = capsys.readouterr()
    
	assert 1 <= elapsed < 1.5
	assert len(captured.out) > 0

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Demo loading animation")
	parser.add_argument(
		"duration",
		type=float,
		nargs="?",  # Makes it optional
		default=2,
		help="Duration in seconds (default: 2)"
	)

	args = parser.parse_args()

	print(f"Start loading animation for {args.duration} seconds")
	start = time.time()

	with loading():
		time.sleep(args.duration)
	
	elapsed = time.time() - start
	print(f"Finished animation, total {elapsed} seconds")