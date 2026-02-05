#	@file: train.py
#	@brief: trains the YOLO model in Tempest

from ultralytics import YOLO
from roboflow import Roboflow

# load local data
DATASET = '/home/v67n884/tempestTraining/Data/dataset-full-mixed-3/data.yaml' # absolute path to data.yaml
# local path to save run to
SAVE_TO = '/home/v67n884/tempestTraining/Results/Full-Mixed-3'

if __name__ == "__main__":
    model = YOLO("yolov8m.pt")
    save_dir = '/home/v67n884/RobosubCV/results'
    results = model.train(data=DATASET,
                          epochs=100,
                          imgsz=640,
                          patience=15,
                          cache=False,
                          seed=17,
                          device=[0,1,2,3],
                          project=SAVE_TO,
                          )