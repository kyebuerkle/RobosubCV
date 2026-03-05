#		live-stream

This directory holds scripts used to showcase the repository. As it can make a GUI that live-streams a model and pints out detected objects.

##		Usage

`yolo_streamer.py` opens the GUI. Follow the steps to use with your selected model (a model is .pt file, I recomend using the 'best.pt' file in your 'weights' directory of your saved YOLO results).

Run: `poetry run python yolo_streamer.py`

The *YOLO Live Stream - CPU Edition* GUI pops up with a blank screen. On the left there are a variety of settings to configure.
- **Model** : Required, the yolo model you wish to test
  - Press the 'Browse .pt file...' button and select the .pt model you wish to use from your device
- **Camera** : Required, the camera you wish to use 
  - Select the drop down menu, and select your camera of choice
- **Resolution** : Optional, the Resolution of the video
- **Confidence threshold** : Optional, the confidence of the model
  - Greater values is less labels (the model needs to be confident to label)
- **Inference size (imgsz)** : Optional, imgsz value for the model
  - Image size to run model on (imgsz does internally)
- **Infer every N frames** : Optional, fraction of frames to run the model on
  - 1 is every frame, 2 is every other frame, 3 is once every 3 frames, ect...
- **Inference Backend** : Optional, the backend for your computer to process the images more efficiently
  - Select one of the three options if they are installed
  - To install the other two, close out and run these commands, open back up to select which to use:
  ```
	pip install onnxruntime
	pip install openvino
  ```
- **OpenCV CPU Threads** : Optional, number of threads OpenCV utilizes
  - This GUI runs on 3 threads already, this number is the amount OpenCV is alowed to use to show the image
  - Read the note on how to decide this, you don't want too much overhead
- **Stream** : Required, runs the stream
  - Press play to play the stream, press stop to stop it
  - Recommended to stop and play everytie you adjust settings
- **Performance stats** : Visual, used to check the stream performance
- **CPU tips** : Visual, tips from Claude on how to optimize performance based on your device
  
---

There is another script, `gpu_yolo_streamer.py`. This is the *GPU Edition* of the application. Use this script if your computer has a NVIDIA GPU to run even faster frame rates. 

##		How To Use depending on Your OS

To use this software, you need access to the cameras and/or USB drives to properly stream a video. To do this you can't run this script on Tempest or on a WSL.
You can run this on a VM or directly off Windows or Linux (not tested on Mac).

Here is a markdown document that explains Windows settup:
![windows_usage.md](../../windows_usage.md)