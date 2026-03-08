#!/bin/bash

# ============================================================
# upload_model.sh
# Transfers a YOLOv8 model to the Jetson Nano over Ethernet
# and triggers the Jetson-side model loader.
#
# Usage:
#     ./upload_model.sh <yolov8m.pt>
#
# Requirements:
#   - SSH key-based login configured for the Jetson
#   - Jetson-side loader script located at:
#         /home/robocats/load_model.py
#   - Target directory:
#         /home/robocats/models/current.pt
# ============================================================

# ---- Input Validation --------------------------------------------------------

if [ -z "$1" ]; then
    echo "ERROR: No model file provided."
    echo "Usage: ./upload_model.sh <yolov8m.pt>"
    exit 1
fi

MODEL_PATH="$1"

if [ ! -f "$MODEL_PATH" ]; then
    echo "ERROR: File '$MODEL_PATH' does not exist."
    exit 1
fi

# ---- Jetson Connection Settings ---------------------------------------------

JETSON_USER="robocats"
JETSON_IP="192.168.1.50"
JETSON_TARGET="$JETSON_USER@$JETSON_IP"
REMOTE_MODEL_PATH="/home/robocats/models/current.pt"
REMOTE_LOADER="/home/robocats/load_model.py"

echo "------------------------------------------------------------"
echo " Robocats Model Upload Script"
echo " Uploading: $MODEL_PATH"
echo " Target:    $JETSON_TARGET"
echo "------------------------------------------------------------"

# ---- Transfer Model ----------------------------------------------------------

echo "[1/3] Transferring model to Jetson Nano..."
scp "$MODEL_PATH" "$JETSON_TARGET:$REMOTE_MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Model transfer failed."
    exit 1
fi

# ---- Trigger Jetson Loader ---------------------------------------------------

echo "[2/3] Running model loader on Jetson..."
ssh "$JETSON_TARGET" "python3 $REMOTE_LOADER $REMOTE_MODEL_PATH"

if [ $? -ne 0 ]; then
    echo "ERROR: Jetson model loader reported a failure."
    exit 1
fi

# ---- Completion --------------------------------------------------------------

echo "[3/3] Upload complete. Jetson has loaded and verified the model."
echo "------------------------------------------------------------"
echo " Model Ready."
echo "------------------------------------------------------------"

exit 0