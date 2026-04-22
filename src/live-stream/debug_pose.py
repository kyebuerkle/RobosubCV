#!/usr/bin/env python3
"""
debug_pose.py — Run this once to diagnose why keypoints aren't rendering.
Usage:  python debug_pose.py path/to/your_model.pt
It opens your webcam for 5 seconds and prints everything about the results.
"""
import sys, time
import cv2
import numpy as np
from ultralytics import YOLO

model_path = sys.argv[1] if len(sys.argv) > 1 else input("Path to .pt model: ").strip()
model = YOLO(model_path)

print(f"\nModel type   : {model.task}")
print(f"Model names  : {model.names}")

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

print("\nCapturing 1 frame...\n")
for _ in range(10):          # skip a few frames for camera to settle
    cap.read()

ret, frame = cap.read()
cap.release()

if not ret:
    print("ERROR: Could not read from camera 0")
    sys.exit(1)

infer = cv2.resize(frame, (320, 320))
results = model(infer, verbose=True, conf=0.1, device="cpu", imgsz=320)

print(f"\n=== Results (len={len(results)}) ===")
for i, r in enumerate(results):
    print(f"\n--- Result[{i}] ---")
    print(f"  r.boxes        : {r.boxes}")
    print(f"  r.keypoints    : {r.keypoints}")

    if r.boxes is not None:
        print(f"  boxes.shape    : {r.boxes.data.shape}")
        print(f"  boxes.data     : {r.boxes.data}")

    if r.keypoints is not None:
        kp = r.keypoints.data
        print(f"  keypoints.shape: {kp.shape}")
        print(f"  keypoints.data :\n{kp}")
        print(f"  --- non-zero kp confs ---")
        for inst in range(kp.shape[0]):
            for k in range(kp.shape[1]):
                x, y, c = float(kp[inst,k,0]), float(kp[inst,k,1]), float(kp[inst,k,2])
                if c > 0.1:
                    print(f"    inst={inst} kp={k:2d}  x={x:.1f} y={y:.1f} conf={c:.3f}")
    else:
        print("  keypoints      : None  <-- THIS IS THE PROBLEM if boxes are found")

print("\n=== Done ===")
