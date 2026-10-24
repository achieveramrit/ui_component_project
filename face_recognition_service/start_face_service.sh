#!/usr/bin/env bash
# Startup script for Face Recognition Service
set -e

SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

echo "=========================================================="
echo " Starting Face Recognition Service (InsightFace + RealSense) "
echo "=========================================================="

# Use virtual environment if present, otherwise system python3
if [ -f "$SCRIPT_DIR/venv/bin/python3" ]; then
    echo "[INFO] Using virtual environment Python ($SCRIPT_DIR/venv)"
    exec "$SCRIPT_DIR/venv/bin/python3" face_server.py "$@"
elif [ -f "$SCRIPT_DIR/.venv/bin/python3" ]; then
    echo "[INFO] Using virtual environment Python ($SCRIPT_DIR/.venv)"
    exec "$SCRIPT_DIR/.venv/bin/python3" face_server.py "$@"
else
    echo "[INFO] Using system python3: $(which python3)"
    exec python3 face_server.py "$@"
fi
