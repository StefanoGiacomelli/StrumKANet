#!/bin/bash

# TensorBoard launcher on Firefox browser

LOG_DIR="./experiments/logs"
PORT=6006
URL="http://localhost:$PORT/"

echo "[INFO] TensorBoard starting on $URL"
tensorboard --logdir "$LOG_DIR" --port "$PORT" --reload_multifile true &

# Wait a few seconds to ensure TensorBoard is up and running
sleep 3

# Open the URL in Firefox if available
if command -v firefox &> /dev/null
then
    echo "[INFO] Opening Firefox on: $URL"
    firefox "$URL" &
else
    echo "[WARNING] Firefox not found. To monitor logger reach: $URL"
fi