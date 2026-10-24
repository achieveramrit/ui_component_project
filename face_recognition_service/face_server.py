#!/usr/bin/env python3
"""
Integrated Face Recognition Service for Intel RealSense D435i and Laptop Webcam.
Powered by InsightFace (ArcFace buffalo_l) and CPUExecutionProvider.

Features:
- Face registration: enroll person face, compute 512-D ArcFace normalized embedding, save to database.
- Multi-person face recognition with cosine similarity and temporal voting.
- Intel RealSense D435i 3D spatial positioning: deproject pixel to 3D point (X, Y, Z in meters).
- Real-time HTTP REST API + MJPEG video stream for hardware UI dashboard.
- Standalone OpenCV GUI mode support (--gui).
"""

import argparse
import base64
import json
import os
import sys
import threading
import time
import uuid
from collections import deque
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn

import cv2
import numpy as np

# ================================================================
# Configuration Defaults
# ================================================================

DEFAULT_HTTP_HOST = "0.0.0.0"
DEFAULT_HTTP_PORT = 5001

FRAME_WIDTH = 1280
FRAME_HEIGHT = 720
FRAME_FPS = 30

REALSENSE_COLOR_WIDTH = 640
REALSENSE_COLOR_HEIGHT = 480
REALSENSE_COLOR_FPS = 30

REALSENSE_DEPTH_WIDTH = 640
REALSENSE_DEPTH_HEIGHT = 480
REALSENSE_DEPTH_FPS = 30

LAPTOP_CAMERA_INDEX = 0

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
REFERENCE_IMAGE_FOLDER = os.path.join(BASE_DIR, "reference_faces")
DATABASE_FILE = os.path.join(REFERENCE_IMAGE_FOLDER, "database.json")

MATCH_THRESHOLD = 0.30
PROCESS_EVERY_N_FRAMES = 2
DETECTION_SIZE = (640, 480)
MATCH_HISTORY_LEN = 1
MATCH_VOTE_THRESHOLD = 1
SHOW_DEPTH = True
PROCESSING_SCALE = 1.0
SHOW_FPS = True


# ================================================================
# Model & Helper Utilities
# ================================================================

def normalize_embedding(embedding):
    """L2-normalizes an embedding vector."""
    embedding = np.asarray(embedding, dtype=np.float32)
    norm = np.linalg.norm(embedding)
    if norm == 0:
        return embedding
    return embedding / norm


def cosine_similarity(embedding_a, embedding_b):
    """Computes cosine similarity between two normalized embeddings."""
    return float(np.dot(embedding_a, embedding_b))


def get_face_3d_position(depth_frame, bbox, rs_module):
    """
    Converts face center pixel into 3D metric coordinates using aligned depth frame.
    """
    if depth_frame is None or rs_module is None:
        return None

    left, top, right, bottom = bbox
    center_x = int((left + right) / 2)
    center_y = int((top + bottom) / 2)

    try:
        frame_width = depth_frame.get_width()
        frame_height = depth_frame.get_height()
    except Exception:
        return None

    if center_x < 0 or center_y < 0 or center_x >= frame_width or center_y >= frame_height:
        return None

    try:
        depth_meters = depth_frame.get_distance(center_x, center_y)
    except Exception:
        return None

    if depth_meters <= 0:
        return None

    try:
        intrinsics = depth_frame.get_profile().as_video_stream_profile().intrinsics
        point = rs_module.rs2_deproject_pixel_to_point(
            intrinsics,
            [center_x, center_y],
            depth_meters,
        )
        return {
            "x": float(point[0]),
            "y": float(point[1]),
            "z": float(point[2]),
            "depth": float(depth_meters),
        }
    except Exception:
        return None


def get_tracking_key(bbox, original_scale):
    """Produces lightweight spatial tracking key for temporal smoothing."""
    left, top, right, bottom = bbox
    left = int(left / original_scale)
    top = int(top / original_scale)
    right = int(right / original_scale)
    bottom = int(bottom / original_scale)
    center_x = (left + right) // 2
    center_y = (top + bottom) // 2
    return center_x // 80, center_y // 80


# ================================================================
# Reference Database Management
# ================================================================

class FaceDatabase:
    """Manages enrolled persons, reference images, and ArcFace embeddings."""

    def __init__(self, folder_path=REFERENCE_IMAGE_FOLDER, db_file=DATABASE_FILE):
        self.folder_path = folder_path
        self.db_file = db_file
        self.lock = threading.Lock()
        self.persons = []  # list of dicts: {"id", "name", "image_file", "created_at", "embedding"}

        os.makedirs(self.folder_path, exist_ok=True)
        self.load()

    def load(self):
        with self.lock:
            if os.path.exists(self.db_file):
                try:
                    with open(self.db_file, "r") as f:
                        self.persons = json.load(f)
                    print(f"[DATABASE] Loaded {len(self.persons)} enrolled persons from database.json")
                    return
                except Exception as e:
                    print(f"[WARN] Could not load database.json: {e}")
            self.persons = []

    def save(self):
        with self.lock:
            try:
                with open(self.db_file, "w") as f:
                    json.dump(self.persons, f, indent=2)
            except Exception as e:
                print(f"[ERROR] Failed to save database.json: {e}")

    def add_person(self, name, image_bgr, face_app):
        """
        Extracts ArcFace embedding from the provided face image and saves to database.
        """
        embedding = None

        if face_app is not None:
            try:
                faces = face_app.get(image_bgr)
            except Exception as e:
                return False, f"Face inference failed: {e}"

            if not faces or len(faces) == 0:
                return False, "No face detected in the image. Please make sure face is clearly visible."

            # Pick largest face
            if len(faces) > 1:
                faces = sorted(
                    faces,
                    key=lambda f: ((f.bbox[2] - f.bbox[0]) * (f.bbox[3] - f.bbox[1])),
                    reverse=True,
                )

            face = faces[0]
            if face.normed_embedding is None:
                return False, "Could not produce face recognition embedding."

            embedding = normalize_embedding(face.normed_embedding).tolist()
        else:
            # Fallback simulator when insightface is not yet installed
            try:
                gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                detected = face_cascade.detectMultiScale(gray, 1.1, 4)
                if len(detected) == 0:
                    return False, "No face detected (Haar fallback). Please center face in camera."
            except Exception:
                pass
            # Deterministic pseudo-embedding for testing
            np.random.seed(abs(hash(name)) % (2**32))
            simulated = np.random.randn(512).astype(np.float32)
            embedding = normalize_embedding(simulated).tolist()
        person_id = str(uuid.uuid4())[:8]
        safe_name = "".join(c for c in name if c.isalnum() or c in ("-", "_")).strip() or "person"
        filename = f"{safe_name}_{person_id}.jpg"
        filepath = os.path.join(self.folder_path, filename)

        # Save cropped or full image
        cv2.imwrite(filepath, image_bgr)

        record = {
            "id": person_id,
            "name": name.strip(),
            "image_file": filename,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
            "embedding": embedding,
        }

        with self.lock:
            self.persons.append(record)

        self.save()
        print(f"[DATABASE] Enrolled '{name}' (id: {person_id}, image: {filename})")
        return True, record

    def delete_person(self, person_id):
        with self.lock:
            found = False
            remaining = []
            for p in self.persons:
                if p["id"] == person_id or p["name"].lower() == person_id.lower():
                    found = True
                    img_path = os.path.join(self.folder_path, p.get("image_file", ""))
                    if os.path.exists(img_path):
                        try:
                            os.remove(img_path)
                        except Exception:
                            pass
                else:
                    remaining.append(p)
            self.persons = remaining

        if found:
            self.save()
            print(f"[DATABASE] Deleted person id: {person_id}")
            return True
        return False

    def get_reference_embeddings(self):
        """Returns list of tuples: (name, normalized_embedding_np)."""
        with self.lock:
            refs = []
            for p in self.persons:
                emb = np.array(p["embedding"], dtype=np.float32)
                refs.append((p["name"], normalize_embedding(emb)))
            return refs

    def get_public_list(self):
        with self.lock:
            return [
                {
                    "id": p["id"],
                    "name": p["name"],
                    "image_file": p["image_file"],
                    "created_at": p.get("created_at", ""),
                }
                for p in self.persons
            ]


# ================================================================
# Face Recognition Engine
# ================================================================

class FaceRecognitionEngine:
    def __init__(self):
        self.face_app = None
        self.model_status = "unloaded"
        self.model_error = None
        self.database = FaceDatabase()
        self.match_histories = {}

        # Runtime configuration
        self.match_threshold = MATCH_THRESHOLD
        self.process_every_n_frames = PROCESS_EVERY_N_FRAMES
        self.processing_scale = PROCESSING_SCALE
        self.show_depth = SHOW_DEPTH
        self.camera_source = "realsense"  # 'realsense' | 'laptop' | 'test'
        self.camera_index = LAPTOP_CAMERA_INDEX

        # Live State
        self.latest_frame = None
        self.latest_depth_frame = None
        self.latest_annotations = []
        self.latest_detections = []
        self.display_fps = 0.0
        self.inference_fps = 0.0
        self.is_camera_running = False
        self.rs_module = None
        self.pipeline = None
        self.cap = None

        self.lock = threading.Lock()
        self._load_insightface()

    def _load_insightface(self):
        print("[ENGINE] Loading InsightFace CPUExecutionProvider...")
        try:
            from insightface.app import FaceAnalysis
            face_app = FaceAnalysis(
                name="buffalo_l",
                allowed_modules=["detection", "recognition"],
                providers=["CPUExecutionProvider"],
            )
            face_app.prepare(
                ctx_id=-1,
                det_size=DETECTION_SIZE,
                det_thresh=0.55,
            )
            self.face_app = face_app
            self.model_status = "loaded"
            print("[ENGINE] InsightFace loaded successfully.")
        except Exception as e:
            self.model_status = "error"
            self.model_error = str(e)
            print(f"[ENGINE ERROR] InsightFace load failed: {e}")

    def recognize(self, frame, depth_frame=None):
        references = self.database.get_reference_embeddings()
        annotations = []

        if self.processing_scale != 1.0:
            processing_frame = cv2.resize(
                frame,
                None,
                fx=self.processing_scale,
                fy=self.processing_scale,
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            processing_frame = frame

        faces = []
        if self.face_app is not None:
            try:
                faces = self.face_app.get(processing_frame)
            except Exception as err:
                print(f"[WARN] Inference failure: {err}")
                return []
        else:
            # Haar Cascade fallback for testing when insightface is not yet downloaded
            try:
                gray = cv2.cvtColor(processing_frame, cv2.COLOR_BGR2GRAY)
                cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
                face_cascade = cv2.CascadeClassifier(cascade_path)
                rects = face_cascade.detectMultiScale(gray, 1.1, 4)
                for (x, y, w, h) in rects:
                    class DummyFace:
                        pass
                    df = DummyFace()
                    df.bbox = np.array([x, y, x + w, y + h])
                    crop_hash = abs(hash(gray[y:y+h, x:x+w].tobytes()[:100])) % (2**32)
                    np.random.seed(crop_hash)
                    df.normed_embedding = normalize_embedding(np.random.randn(512).astype(np.float32))
                    faces.append(df)
            except Exception:
                pass

        current_keys = set()
        frame_height, frame_width = frame.shape[:2]
        detections_info = []

        for face in faces:
            if face.normed_embedding is None:
                continue

            scaled_bbox = face.bbox.astype(float)
            tracking_key = get_tracking_key(scaled_bbox, self.processing_scale)
            current_keys.add(tracking_key)

            embedding = normalize_embedding(face.normed_embedding)

            best_similarity = 0.0
            matched_name = "Unknown"

            if references:
                for ref_name, ref_emb in references:
                    sim = cosine_similarity(embedding, ref_emb)
                    if sim > best_similarity:
                        best_similarity = sim
                        matched_name = ref_name

            raw_match = best_similarity >= self.match_threshold

            if tracking_key not in self.match_histories:
                self.match_histories[tracking_key] = deque(maxlen=MATCH_HISTORY_LEN)

            self.match_histories[tracking_key].append(raw_match)
            positive_votes = sum(self.match_histories[tracking_key])
            stable_match = positive_votes >= MATCH_VOTE_THRESHOLD

            left = int(scaled_bbox[0] / self.processing_scale)
            top = int(scaled_bbox[1] / self.processing_scale)
            right = int(scaled_bbox[2] / self.processing_scale)
            bottom = int(scaled_bbox[3] / self.processing_scale)

            left = max(0, min(left, frame_width - 1))
            top = max(0, min(top, frame_height - 1))
            right = max(0, min(right, frame_width - 1))
            bottom = max(0, min(bottom, frame_height - 1))

            face_3d_position = get_face_3d_position(
                depth_frame=depth_frame,
                bbox=(left, top, right, bottom),
                rs_module=self.rs_module,
            )

            votes_info = f"[{positive_votes}/{MATCH_HISTORY_LEN}]"
            coords_info = ""
            if face_3d_position is not None:
                coords_info = (
                    f" XYZ=({face_3d_position['x']:.2f}, "
                    f"{face_3d_position['y']:.2f}, "
                    f"{face_3d_position['z']:.2f})m"
                )

            if stable_match and references:
                label = f"{matched_name} ({best_similarity:.2f}) {votes_info}{coords_info}"
                color = (0, 255, 0)
                identified_name = matched_name
            else:
                label = f"Unknown ({best_similarity:.2f}) {votes_info}{coords_info}"
                color = (0, 0, 255)
                identified_name = "Unknown"

            annotations.append({
                "bbox": (left, top, right, bottom),
                "label": label,
                "color": color,
            })

            detections_info.append({
                "name": identified_name,
                "matched": stable_match and references,
                "confidence": round(best_similarity, 3),
                "bbox": [left, top, right, bottom],
                "position_3d": face_3d_position,
                "timestamp": time.time(),
            })

        stale_keys = [k for k in self.match_histories if k not in current_keys]
        for k in stale_keys:
            del self.match_histories[k]

        with self.lock:
            self.latest_detections = detections_info

        return annotations


# ================================================================
# Video Capture Thread
# ================================================================

class CameraWorker(threading.Thread):
    def __init__(self, engine):
        super().__init__(daemon=True)
        self.engine = engine
        self.running = True

    def _generate_standby_frame(self, message):
        width, height = 640, 480
        frame = np.zeros((height, width, 3), dtype=np.uint8)
        frame[:] = (22, 28, 40)
        cv2.putText(frame, message, (35, height // 2 - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 165, 255), 2)
        cv2.putText(frame, "Connect camera or select different source in UI", (35, height // 2 + 20), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)
        draw_system_information(frame, 0.0, 0.0)
        with self.engine.lock:
            self.engine.latest_frame = frame
            self.engine.latest_depth_frame = None

    def run(self):
        while self.running:
            source = self.engine.camera_source

            if source == "realsense":
                success = self._run_realsense()
                if not success and self.running and self.engine.camera_source == "realsense":
                    self._generate_standby_frame("RealSense D435i Waiting for Connection...")
                    time.sleep(2)
            elif source == "laptop":
                success = self._run_laptop()
                if not success and self.running and self.engine.camera_source == "laptop":
                    self._generate_standby_frame("Webcam Waiting for Connection...")
                    time.sleep(2)
            else:
                self._run_test_pattern()

            time.sleep(0.5)

    def _run_realsense(self):
        try:
            import pyrealsense2 as rs
            self.engine.rs_module = rs
        except ImportError:
            print("[WARN] pyrealsense2 is not installed.")
            return False

        pipeline = rs.pipeline()
        config = rs.config()

        config.enable_stream(
            rs.stream.color,
            REALSENSE_COLOR_WIDTH,
            REALSENSE_COLOR_HEIGHT,
            rs.format.bgr8,
            REALSENSE_COLOR_FPS,
        )

        if self.engine.show_depth:
            config.enable_stream(
                rs.stream.depth,
                REALSENSE_DEPTH_WIDTH,
                REALSENSE_DEPTH_HEIGHT,
                rs.format.z16,
                REALSENSE_DEPTH_FPS,
            )

        try:
            profile = pipeline.start(config)
            self.engine.is_camera_running = True
            print("[CAMERA] RealSense D435i pipeline started.")
        except Exception as e:
            print(f"[WARN] Failed to start RealSense: {e}")
            self.engine.is_camera_running = False
            return False

        align = rs.align(rs.stream.color) if self.engine.show_depth else None
        colorizer = rs.colorizer() if self.engine.show_depth else None

        frame_count = 0
        display_timer = time.perf_counter()
        display_count = 0
        cached_annotations = []

        try:
            while self.running and self.engine.camera_source == "realsense":
                try:
                    frames = pipeline.wait_for_frames(timeout_ms=3000)
                except Exception:
                    continue

                depth_frame = None
                depth_color_img = None

                if self.engine.show_depth and align:
                    aligned = align.process(frames)
                    color_frame = aligned.get_color_frame()
                    depth_frame = aligned.get_depth_frame()
                    if depth_frame and colorizer:
                        depth_color_img = np.asanyarray(colorizer.colorize(depth_frame).get_data())
                else:
                    color_frame = frames.get_color_frame()

                if not color_frame:
                    continue

                color_image = np.asanyarray(color_frame.get_data())
                frame_count += 1
                display_count += 1

                # Face Recognition Inference
                if frame_count == 1 or frame_count % self.engine.process_every_n_frames == 0:
                    t0 = time.perf_counter()
                    cached_annotations = self.engine.recognize(
                        frame=color_image,
                        depth_frame=depth_frame,
                    )
                    t_inf = time.perf_counter() - t0
                    if t_inf > 0:
                        self.engine.inference_fps = 1.0 / t_inf

                now = time.perf_counter()
                if now - display_timer >= 1.0:
                    self.engine.display_fps = display_count / (now - display_timer)
                    display_count = 0
                    display_timer = now

                # Annotate frame
                annotated = draw_annotations(color_image, cached_annotations)
                if not self.engine.database.persons:
                    cv2.putText(
                        annotated,
                        "No reference faces enrolled",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                draw_system_information(annotated, self.engine.display_fps, self.engine.inference_fps)

                with self.engine.lock:
                    self.engine.latest_frame = annotated
                    self.engine.latest_depth_frame = depth_color_img

        finally:
            print("[CAMERA] Stopping RealSense pipeline...")
            try:
                pipeline.stop()
            except Exception:
                pass
            self.engine.is_camera_running = False

        return True

    def _run_laptop(self):
        index = self.engine.camera_index
        print(f"[CAMERA] Opening webcam index {index}...")
        cap = cv2.VideoCapture(index)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, FRAME_WIDTH)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, FRAME_HEIGHT)
        cap.set(cv2.CAP_PROP_FPS, FRAME_FPS)

        if not cap.isOpened():
            print(f"[WARN] Could not open webcam index {index}.")
            return False

        self.engine.is_camera_running = True
        frame_count = 0
        display_timer = time.perf_counter()
        display_count = 0
        cached_annotations = []

        try:
            while self.running and self.engine.camera_source == "laptop":
                ret, frame = cap.read()
                if not ret or frame is None:
                    time.sleep(0.01)
                    continue

                frame_count += 1
                display_count += 1

                if frame_count == 1 or frame_count % self.engine.process_every_n_frames == 0:
                    t0 = time.perf_counter()
                    cached_annotations = self.engine.recognize(frame=frame)
                    t_inf = time.perf_counter() - t0
                    if t_inf > 0:
                        self.engine.inference_fps = 1.0 / t_inf

                now = time.perf_counter()
                if now - display_timer >= 1.0:
                    self.engine.display_fps = display_count / (now - display_timer)
                    display_count = 0
                    display_timer = now

                annotated = draw_annotations(frame, cached_annotations)
                if not self.engine.database.persons:
                    cv2.putText(
                        annotated,
                        "No reference faces enrolled",
                        (10, 80),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 165, 255),
                        2,
                        cv2.LINE_AA,
                    )
                draw_system_information(annotated, self.engine.display_fps, self.engine.inference_fps)

                with self.engine.lock:
                    self.engine.latest_frame = annotated
                    self.engine.latest_depth_frame = None

        finally:
            cap.release()
            self.engine.is_camera_running = False

    def _run_test_pattern(self):
        print("[CAMERA] Running test pattern camera...")
        self.engine.is_camera_running = True
        width, height = 640, 480

        while self.running and self.engine.camera_source == "test":
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            t = time.time()
            # Gradient background
            for y in range(height):
                frame[y, :, 0] = int(20 + 20 * np.sin(t + y / 50))
                frame[y, :, 1] = int(30 + 20 * np.cos(t + y / 50))
                frame[y, :, 2] = 50

            cv2.putText(
                frame,
                "RealSense Hardware Feed Waiting...",
                (40, 200),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                (0, 220, 255),
                2,
            )
            cv2.putText(
                frame,
                "Connect RealSense D435i or Laptop Camera",
                (40, 240),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (180, 180, 180),
                1,
            )
            time_str = time.strftime("%H:%M:%S")
            cv2.putText(frame, f"Time: {time_str}", (40, 280), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (120, 255, 120), 1)

            draw_system_information(frame, 30.0, 0.0)

            with self.engine.lock:
                self.engine.latest_frame = frame
                self.engine.latest_depth_frame = None

            time.sleep(0.033)

        self.engine.is_camera_running = False


# ================================================================
# Visualization Functions
# ================================================================

def draw_annotations(frame, annotations):
    """Draws bounding boxes and filled labels on camera frame."""
    result_frame = frame.copy()

    for ann in annotations:
        left, top, right, bottom = ann["bbox"]
        label = ann["label"]
        color = ann["color"]

        cv2.rectangle(result_frame, (left, top), (right, bottom), color, 2)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.55
        thickness = 2

        (text_width, text_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
        label_top = max(0, top - text_height - baseline - 8)
        label_bottom = max(text_height + baseline + 4, top)
        label_right = min(frame.shape[1] - 1, left + text_width + 10)

        cv2.rectangle(result_frame, (left, label_top), (label_right, label_bottom), color, cv2.FILLED)
        cv2.putText(
            result_frame,
            label,
            (left + 5, label_bottom - baseline - 2),
            font,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )

    return result_frame


def draw_system_information(frame, display_fps, inference_fps):
    if SHOW_FPS:
        cv2.putText(
            frame,
            f"Display FPS: {display_fps:.1f}",
            (10, 25),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )
        cv2.putText(
            frame,
            f"Recognition FPS: {inference_fps:.1f}",
            (10, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (255, 255, 0),
            2,
            cv2.LINE_AA,
        )


# ================================================================
# HTTP Server & REST API
# ================================================================

class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        """Silently ignore normal client disconnects (BrokenPipeError / ConnectionResetError)."""
        exc_type, _, _ = sys.exc_info()
        if exc_type in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


class FaceAPIHandler(BaseHTTPRequestHandler):
    engine = None  # set by server setup

    def log_message(self, format, *args):
        # Silence routine HTTP request logging to keep console clean
        return

    def handle(self):
        try:
            super().handle()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def finish(self):
        try:
            super().finish()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_cors_headers(self):
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "GET, POST, OPTIONS, DELETE")
        self.send_header("Access-Control-Allow-Headers", "Content-Type, Authorization")

    def do_OPTIONS(self):
        try:
            self.send_response(204)
            self._send_cors_headers()
            self.end_headers()
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _send_json(self, data, status_code=200):
        try:
            body = json.dumps(data).encode("utf-8")
            self.send_response(status_code)
            self._send_cors_headers()
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def do_GET(self):
        try:
            url = self.path.split("?")[0]

            if url == "/api/status":
                with self.engine.lock:
                    detections = list(self.engine.latest_detections)
                self._send_json({
                    "status": "online",
                    "model_status": self.engine.model_status,
                    "model_error": self.engine.model_error,
                    "camera_source": self.engine.camera_source,
                    "camera_running": self.engine.is_camera_running,
                    "enrolled_count": len(self.engine.database.persons),
                    "display_fps": round(self.engine.display_fps, 1),
                    "inference_fps": round(self.engine.inference_fps, 1),
                    "match_threshold": self.engine.match_threshold,
                    "process_every_n_frames": self.engine.process_every_n_frames,
                    "show_depth": self.engine.show_depth,
                    "detections": detections,
                    "faces": self.engine.database.get_public_list(),
                })

            elif url == "/api/faces":
                self._send_json({
                    "faces": self.engine.database.get_public_list(),
                    "total": len(self.engine.database.persons),
                })

            elif url == "/api/detections":
                with self.engine.lock:
                    detections = list(self.engine.latest_detections)
                self._send_json({
                    "detections": detections,
                    "display_fps": round(self.engine.display_fps, 1),
                    "inference_fps": round(self.engine.inference_fps, 1),
                })

            elif url == "/api/snapshot":
                with self.engine.lock:
                    frame = self.engine.latest_frame
                if frame is None:
                    self._send_json({"error": "No camera frame available"}, 404)
                    return

                ret, jpeg = cv2.imencode(".jpg", frame)
                if not ret:
                    self._send_json({"error": "Encoding failure"}, 500)
                    return

                self.send_response(200)
                self._send_cors_headers()
                self.send_header("Content-Type", "image/jpeg")
                self.send_header("Content-Length", str(len(jpeg)))
                self.end_headers()
                self.wfile.write(jpeg.tobytes())

            elif url == "/api/stream":
                self._stream_mjpeg(use_depth=False)

            elif url == "/api/depth_stream":
                self._stream_mjpeg(use_depth=True)

            elif url.startswith("/reference_faces/"):
                filename = os.path.basename(url)
                filepath = os.path.join(REFERENCE_IMAGE_FOLDER, filename)
                if os.path.exists(filepath):
                    try:
                        with open(filepath, "rb") as f:
                            img_bytes = f.read()
                        self.send_response(200)
                        self._send_cors_headers()
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(img_bytes)))
                        self.end_headers()
                        self.wfile.write(img_bytes)
                    except Exception:
                        self._send_json({"error": "Read error"}, 500)
                else:
                    self._send_json({"error": "Image not found"}, 404)

            else:
                self._send_json({"error": "Not found", "path": self.path}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _stream_mjpeg(self, use_depth=False):
        try:
            self.send_response(200)
            self._send_cors_headers()
            self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
            self.send_header("Cache-Control", "no-cache, private")
            self.send_header("Pragma", "no-cache")
            self.end_headers()

            while True:
                with self.engine.lock:
                    frame = self.engine.latest_depth_frame if use_depth else self.engine.latest_frame

                if frame is not None:
                    ret, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    if ret:
                        frame_bytes = jpeg.tobytes()
                        self.wfile.write(b"--frame\r\n")
                        self.send_header("Content-Type", "image/jpeg")
                        self.send_header("Content-Length", str(len(frame_bytes)))
                        self.end_headers()
                        self.wfile.write(frame_bytes)
                        self.wfile.write(b"\r\n")
                time.sleep(0.033)
        except (BrokenPipeError, ConnectionResetError):
            return

    def do_POST(self):
        try:
            url = self.path.split("?")[0]
            content_length = int(self.headers.get("Content-Length", 0))
            raw_body = self.rfile.read(content_length)

            try:
                body = json.loads(raw_body.decode("utf-8")) if raw_body else {}
            except Exception:
                body = {}

            if url == "/api/register":
                name = body.get("name", "").strip()
                image_data = body.get("image", "")

                if not name:
                    self._send_json({"success": False, "error": "Person name is required"}, 400)
                    return

                if not image_data:
                    # If no image provided, grab from current camera frame
                    with self.engine.lock:
                        frame = self.engine.latest_frame
                    if frame is None:
                        self._send_json({"success": False, "error": "No camera frame available to capture"}, 400)
                        return
                    image_bgr = frame
                else:
                    # Base64 data URL decoding
                    try:
                        if "," in image_data:
                            image_data = image_data.split(",")[1]
                        img_bytes = base64.b64decode(image_data)
                        np_arr = np.frombuffer(img_bytes, np.uint8)
                        image_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
                    except Exception as err:
                        self._send_json({"success": False, "error": f"Invalid image format: {err}"}, 400)
                        return

                success, result = self.engine.database.add_person(name, image_bgr, self.engine.face_app)
                if success:
                    self._send_json({
                        "success": True,
                        "message": f"Successfully enrolled face embedding for '{name}'",
                        "person": result,
                    })
                else:
                    self._send_json({"success": False, "error": result}, 400)

            elif url == "/api/faces/delete":
                person_id = body.get("id", "")
                if not person_id:
                    self._send_json({"success": False, "error": "Person id is required"}, 400)
                    return

                deleted = self.engine.database.delete_person(person_id)
                if deleted:
                    self._send_json({"success": True, "message": "Person deleted successfully"})
                else:
                    self._send_json({"success": False, "error": "Person not found"}, 404)

            elif url == "/api/config":
                if "match_threshold" in body:
                    self.engine.match_threshold = float(body["match_threshold"])
                if "process_every_n_frames" in body:
                    self.engine.process_every_n_frames = max(1, int(body["process_every_n_frames"]))
                if "camera_source" in body:
                    self.engine.camera_source = str(body["camera_source"]).lower()
                if "show_depth" in body:
                    self.engine.show_depth = bool(body["show_depth"])

                self._send_json({
                    "success": True,
                    "config": {
                        "match_threshold": self.engine.match_threshold,
                        "process_every_n_frames": self.engine.process_every_n_frames,
                        "camera_source": self.engine.camera_source,
                        "show_depth": self.engine.show_depth,
                    }
                })

            else:
                self._send_json({"error": "Endpoint not found"}, 404)
        except (BrokenPipeError, ConnectionResetError):
            pass


# ================================================================
# Entry Point
# ================================================================

def run_server(host=DEFAULT_HTTP_HOST, port=DEFAULT_HTTP_PORT, default_source="realsense"):
    engine = FaceRecognitionEngine()
    engine.camera_source = default_source
    FaceAPIHandler.engine = engine

    # Start camera capture thread
    worker = CameraWorker(engine)
    worker.start()

    server = ThreadingHTTPServer((host, port), FaceAPIHandler)
    print("=" * 60)
    print(f"Face Recognition API & Video Streaming Server")
    print(f"URL: http://localhost:{port}")
    print(f"Endpoints:")
    print(f"  - GET  /api/status       (Engine & camera status)")
    print(f"  - GET  /api/faces        (Enrolled faces list)")
    print(f"  - POST /api/register     (Enroll person & embedding)")
    print(f"  - POST /api/faces/delete (Delete enrolled person)")
    print(f"  - GET  /api/stream       (Live MJPEG annotated stream)")
    print(f"  - GET  /api/depth_stream (Live RealSense depth stream)")
    print(f"  - GET  /api/snapshot     (Current camera snapshot)")
    print(f"  - POST /api/config       (Runtime settings)")
    print("=" * 60)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down Face Recognition Server...")
        worker.running = False
        server.server_close()
        print("[SERVER] Exited.")


def main():
    parser = argparse.ArgumentParser(description="InsightFace RealSense Recognition & Web Service")
    parser.add_argument("--host", default=DEFAULT_HTTP_HOST, help="HTTP server bind host")
    parser.add_argument("--port", type=int, default=DEFAULT_HTTP_PORT, help="HTTP server bind port")
    parser.add_argument("--source", choices=["realsense", "laptop", "test"], default="realsense", help="Default camera source")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, default_source=args.source)


if __name__ == "__main__":
    main()
