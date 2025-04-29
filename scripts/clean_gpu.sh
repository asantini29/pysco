#!/bin/bash

# Check if a GPU ID is provided
if [ -z "$1" ]; then
    echo "Usage: $0 <GPU_ID>"
    exit 1
fi

GPU_ID=$1

# Get the PIDs of all Python processes running on the specified GPU
PIDS=$(nvidia-smi -i "$GPU_ID" --query-compute-apps=pid,process_name,gpu_uuid --format=csv,noheader | grep python | cut -d, -f1)

# Check if any Python processes are running
if [ -z "$PIDS" ]; then
    echo "No Python processes running on GPU $GPU_ID."
    exit 0
fi

# Kill each Python process
echo "Killing Python processes on GPU $GPU_ID..."
for PID in $PIDS; do
    echo "Killing process $PID"
    kill -9 $PID
done

echo "Done."