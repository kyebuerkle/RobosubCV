#		live-stream

This directory holds scripts used to showcase the repository. As it can make a GUI that live-streams a model and pints out detected objects.

##		Usage

`yolo_streamer.py` opens the GUI. Follow the steps to use with your selected model (a model is .pt file, I recomend using the 'best.pt' file in your 'weights' directory of your saved YOLO results).

Run: `poetry run python yolo_streamer.py`

To use this first select your .pt file. To pick a file press the top left button that asks for a file.
Next select your camera. Right below the model select there should be a drop down button that lets you select a given camera.
Press 'Stream' to start streaming!

NOTE: currently this only works at below 5 FPS, which is below our spec. But this is just a showcase, not a requirment for the project. The submarine has around 10 FPS.

##		How To Use depending on Your OS

To use this software, you need access to the cameras and/or USB drives to properly stream a video. To do this you can't run this script on Tempest or on a WSL.
You can run this on a VM or directly off Windows or Linux (not tested on Mac).

Here is a markdown document that explains Windows settup:
![windows_usage.md](../../windows_usage.md)