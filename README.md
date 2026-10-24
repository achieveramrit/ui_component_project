# Autonomous Ground Vehicle (AGV) Dashboard & Perception Hub

An integrated, dual-tab hardware control and vision dashboard built with **React (Vite)** and **ROS 2**, featuring real-time SLAM mapping, AGV motion controls, and an **InsightFace ArcFace** facial recognition suite with **Intel RealSense D435i** 3D metric spatial position deprojection.

---

## Architecture Overview

```
ui_component_project/
├── face_recognition_service/            # Python Vision & Recognition Service
│   ├── face_server.py                   # Multi-threaded REST API & MJPEG streaming server
│   ├── requirements.txt                 # Python dependencies (InsightFace, OpenCV, etc.)
│   ├── start_face_service.sh            # Auto-detects venv and launches service
│   ├── reference_faces/                 # Stored reference photos and database.json
│   └── README.md                        # Service documentation
├── src/
│   ├── components/
│   │   └── FaceRecognitionTab.jsx       # Dedicated Face Recognition React component
│   ├── App.jsx                          # Main shell, tab switcher, & SLAM dashboard
│   ├── App.css                          # Dark-mode dashboard styling
│   └── main.jsx
├── index.html
├── package.json                         # Node.js dependencies (React 19, roslib v2, Vite)
├── .gitignore                           # Excludes node_modules, venv, dist, and pycache
└── README.md
```

---

## Features

### Tab 1: Robot SLAM & Navigation Dashboard
- **Live Occupancy Grid (`/map`)**: Real-time canvas rendering of 2D SLAM maps.
- **Odometry & Telemetry (`/odom`)**: Metric coordinates ($X$, $Y$), heading orientation in degrees, and linear velocity ($m/s$).
- **Dual Camera Feeds**:
  - Live RGB Camera: `/camera/color/image_raw/compressed`
  - Colorized Metric Depth: `/camera/depth/image_colorized/compressed`
- **Frontier Exploration (`/explore/resume`)**: Start, resume, or pause autonomous exploration.
- **Manual Velocity Control (`/cmd_vel`)**: Directional motion pad with safety zero-velocity stop.

### Tab 2: Face Recognition & RealSense Perception Hub
- **Multi-Person Face Enrollment**:
  - Register any person by name.
  - Choose between **"📷 Snapshot from Camera"** (captures directly from hardware stream) or **"📁 Browse Photo"** (uploads `.jpg`, `.png`, `.webp`).
  - Automatically computes and normalizes 512-dimensional ArcFace (`buffalo_l`) embeddings and stores them in `reference_faces/database.json`.
- **Live Hardware Video Stream**:
  - High-performance MJPEG stream with colored bounding boxes (green for recognized, red for unknown).
  - Displays person name, ArcFace cosine similarity score, and temporal voting status `[votes/total]`.
- **Intel RealSense D435i 3D Spatial Deprojection**:
  - Uses aligned depth frame intrinsics to compute metric real-world coordinates: `XYZ=(x, y, z)m`.
- **Aligned Depth Colormap**:
  - Optional side-by-side stream showing aligned metric depth colormap (`/api/depth_stream`).
- **Independent Tab Refresh ("Refresh Tab")**:
  - Dedicated refresh button tests backend connectivity and updates states without reloading the browser or disconnecting the SLAM tab.
  - Video stream reload button (`🔄 Stream`) to re-establish the camera stream if hardware is replugged.
- **Interactive Controls**:
  - Cosine match threshold slider (`0.10` to `0.80`, default `0.30`).
  - Frame skip selector (run inference every 1, 2, 4, or 8 frames).
  - Camera source selector (`Intel RealSense D435i`, `Laptop/USB Webcam`, or `Simulated Test Stream`).
- **Enrolled Faces Directory**:
  - Gallery cards displaying registered thumbnails, names, 512-D vector indicators, and instant deletion.
- **Real-Time Activity Log**:
  - Live badges for detected faces plus a rolling chronological recognition log.

---

## Getting Started

### 1. Prerequisites
- **Node.js**: v18+ or LTS
- **Python**: 3.10+
- **Hardware**: Intel RealSense D435i (USB 3.0 connection recommended) or standard USB/laptop webcam
- **ROS 2**: Humble Hawksbill with `rosbridge_server` (for SLAM tab)

---

### 2. Setup & Installation

#### Frontend Setup
```bash
cd ui_component_project
npm install
```

#### Backend Setup (Face Recognition Service)
```bash
cd face_recognition_service
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

> **Note**: If using Intel RealSense D435i, ensure `pyrealsense2` is installed:
> ```bash
> pip install pyrealsense2
> ```

---

### 3. Running the System

#### Step 1: Start the Face Recognition Backend
From `ui_component_project/face_recognition_service/`:
```bash
./start_face_service.sh
```
Or with custom flags:
```bash
python3 face_server.py --port 5001 --source realsense
```

Available camera source flags:
- `--source realsense`: Intel RealSense D435i (Color + Aligned 3D Depth)
- `--source laptop`: Laptop or USB webcam
- `--source test`: Simulated test stream (for testing without physical camera)

#### Step 2: Start the Web Dashboard
From `ui_component_project/`:
```bash
npm run dev
```
Open the Vite URL printed in the terminal (default: `http://localhost:5173`) in your web browser.

#### Step 3: Connect to ROS 2 (for SLAM tab)
1. On the robot computer running ROS 2 Humble:
   ```bash
   ros2 launch rosbridge_server rosbridge_websocket_launch.xml
   ```
2. In the dashboard top bar, enter the robot's WebSocket URL (e.g., `ws://192.168.1.50:9090`) and click **Connect**.

---

## Face Recognition REST API Reference

The backend runs on port `5001` (by default) with CORS enabled:

| Method | Endpoint | Description |
|---|---|---|
| `GET` | `/api/status` | Returns backend state, active camera source, model status, FPS, enrolled faces, and current detections. |
| `GET` | `/api/faces` | Returns list of all registered individuals with photo references. |
| `POST` | `/api/register` | Accepts JSON `{ "name": "...", "image": "<base64>" }`, computes 512-D ArcFace embedding, and saves to database. |
| `POST` | `/api/faces/delete` | Accepts JSON `{ "id": "..." }` and deletes the person and embeddings. |
| `GET` | `/api/stream` | Multipart MJPEG stream of annotated camera frames with bounding boxes. |
| `GET` | `/api/depth_stream` | Multipart MJPEG stream of RealSense colorized aligned depth map. |
| `GET` | `/api/snapshot` | Returns a single JPEG frame from the camera (used for enrollment snapshot). |
| `POST` | `/api/config` | Updates runtime configuration (`match_threshold`, `camera_source`, `process_every_n_frames`, `show_depth`). |
| `GET` | `/reference_faces/<file>` | Serves saved face thumbnail images for the dashboard gallery. |

---

## ROS 2 Topics Used

| Topic | Message Type | Purpose |
|---|---|---|
| `/map` | `nav_msgs/OccupancyGrid` | Live 2D SLAM grid map |
| `/odom` | `nav_msgs/Odometry` | Robot position, heading, and speed |
| `/cmd_vel` | `geometry_msgs/Twist` | Velocity commands from manual motion pad |
| `/explore/resume` | `std_msgs/Bool` | Resume (`true`) or pause (`false`) frontier exploration |
| `/camera/color/image_raw/compressed` | `sensor_msgs/CompressedImage` | Compressed color camera feed |
| `/camera/depth/image_colorized/compressed` | `sensor_msgs/CompressedImage` | Compressed colorized depth feed |

---

## Safety Guidelines

Manual velocity controls publish live `/cmd_vel` commands to the robot. Always:
- Test first with drive wheels elevated off the ground.
- Ensure the robot has an operational hardware emergency stop.
- Maintain a safe testing perimeter.
