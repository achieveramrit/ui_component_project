# Face Recognition Service (InsightFace + Intel RealSense D435i)

This service integrates ArcFace face recognition (`buffalo_l`) and Intel RealSense D435i 3D metric spatial position deprojection with the hardware dashboard UI.

## Features
- **Person Face Enrollment**: Upload or capture face images, compute 512-D ArcFace normalized embeddings, and save to `reference_faces/database.json`.
- **Live Frame-by-Frame Recognition**: Performs ArcFace inference, temporal voting `[votes/total]`, and cosine similarity matching.
- **Intel RealSense D435i 3D Deprojection**: Uses the aligned depth frame intrinsics and distance to compute real-world metric coordinates: `XYZ=(x, y, z)m`.
- **Dual Streaming**: Provides real-time MJPEG stream with bounding boxes and an optional aligned colorized depth map stream.
- **REST API**: Clean endpoints for person enrollment, deletion, live metadata, snapshots, and runtime config.

## Requirements & Setup

Install Python packages:
```bash
pip install -r requirements.txt
```

> Note: If using Intel RealSense D435i, ensure `pyrealsense2` is installed and the camera is plugged into a USB 3.0 port.

## Running the Service

Start the service on default port 5001:
```bash
./start_face_service.sh
```
Or specify options:
```bash
python3 face_server.py --port 5001 --source realsense
```

Camera sources:
- `--source realsense` (Default: Intel RealSense D435i color + depth)
- `--source laptop` (Standard USB / Laptop webcam)
- `--source test` (Simulated test feed for testing without camera connected)
