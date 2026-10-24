import React, { useState, useEffect, useRef } from "react";

const DEFAULT_SERVER_URL = "http://localhost:5001";

export default function FaceRecognitionTab() {
  const [serverUrl, setServerUrl] = useState(DEFAULT_SERVER_URL);
  const [connected, setConnected] = useState(false);
  const [status, setStatus] = useState(null);
  const [faces, setFaces] = useState([]);
  const [detections, setDetections] = useState([]);
  const [detectionLogs, setDetectionLogs] = useState([]);

  // Refresh & Reconnect state (reloads tab only, not whole page)
  const [refreshing, setRefreshing] = useState(false);
  const [refreshFeedback, setRefreshFeedback] = useState(null);
  const [streamTimestamp, setStreamTimestamp] = useState(Date.now());

  // Enrollment form state
  const [personName, setPersonName] = useState("");
  const [previewImage, setPreviewImage] = useState(null);
  const [enrollLoading, setEnrollLoading] = useState(false);
  const [enrollFeedback, setEnrollFeedback] = useState(null);
  const fileInputRef = useRef(null);

  // Runtime settings
  const [cameraSource, setCameraSource] = useState("realsense");
  const [matchThreshold, setMatchThreshold] = useState(0.30);
  const [frameSkip, setFrameSkip] = useState(2);
  const [showDepthView, setShowDepthView] = useState(true);

  // Snapshot taking indicator
  const [capturingSnapshot, setCapturingSnapshot] = useState(false);

  // Core status polling function
  async function fetchStatusData(url = serverUrl) {
    const res = await fetch(`${url}/api/status?t=${Date.now()}`);
    if (!res.ok) throw new Error("HTTP error " + res.status);
    return await res.json();
  }

  // Manual refresh button: re-checks connection & reloads recognition view only
  async function handleManualRefresh() {
    setRefreshing(true);
    setRefreshFeedback({ type: "info", text: "Checking Face Recognition backend..." });
    try {
      const data = await fetchStatusData(serverUrl);
      setConnected(true);
      setStatus(data);
      setCameraSource(data.camera_source || "realsense");
      setMatchThreshold(data.match_threshold ?? 0.30);
      setFrameSkip(data.process_every_n_frames ?? 2);
      if (data.faces) setFaces(data.faces);
      if (data.detections) setDetections(data.detections);
      // Update stream key to force re-mounting the MJPEG stream element without full-page reload
      setStreamTimestamp(Date.now());
      setRefreshFeedback({
        type: "success",
        text: `Connected to backend! Camera: ${(data.camera_source || "realsense").toUpperCase()} · Model: ${data.model_status} · ${data.enrolled_count ?? data.faces?.length ?? 0} enrolled faces`,
      });
    } catch (err) {
      setConnected(false);
      setStatus(null);
      setRefreshFeedback({
        type: "error",
        text: `Backend offline at ${serverUrl}. Make sure ./face_recognition_service/start_face_service.sh is running.`,
      });
    } finally {
      setRefreshing(false);
      setTimeout(() => setRefreshFeedback(null), 4500);
    }
  }

  // Periodic background status poll
  useEffect(() => {
    let isMounted = true;
    let timer = null;

    async function poll() {
      try {
        const data = await fetchStatusData(serverUrl);
        if (isMounted) {
          setConnected(true);
          setStatus(data);
          setCameraSource(data.camera_source || "realsense");
          setMatchThreshold(data.match_threshold ?? 0.30);
          setFrameSkip(data.process_every_n_frames ?? 2);
          if (data.faces) setFaces(data.faces);
          if (data.detections) {
            setDetections(data.detections);
            if (data.detections.length > 0) {
              setDetectionLogs((prev) => {
                const newEntries = data.detections.map((d) => ({
                  id: Math.random().toString(36).substring(7),
                  name: d.name,
                  matched: d.matched,
                  confidence: d.confidence,
                  position_3d: d.position_3d,
                  time: new Date().toLocaleTimeString(),
                }));
                return [...newEntries, ...prev].slice(0, 30);
              });
            }
          }
        }
      } catch (err) {
        if (isMounted) {
          setConnected(false);
          setStatus(null);
        }
      } finally {
        if (isMounted) {
          timer = setTimeout(poll, 1500);
        }
      }
    }

    poll();

    return () => {
      isMounted = false;
      if (timer) clearTimeout(timer);
    };
  }, [serverUrl]);

  // Handle configuration changes
  async function updateConfig(newConfig) {
    try {
      await fetch(`${serverUrl}/api/config`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(newConfig),
      });
    } catch (e) {
      console.error("Config update failed:", e);
    }
  }

  function handleCameraSourceChange(source) {
    setCameraSource(source);
    updateConfig({ camera_source: source });
    // Refresh stream timestamp after source switch
    setTimeout(() => setStreamTimestamp(Date.now()), 600);
  }

  function handleThresholdChange(val) {
    const num = parseFloat(val);
    setMatchThreshold(num);
    updateConfig({ match_threshold: num });
  }

  function handleFrameSkipChange(val) {
    const num = parseInt(val, 10);
    setFrameSkip(num);
    updateConfig({ process_every_n_frames: num });
  }

  // Handle file picker selection
  function handleFileSelected(e) {
    const file = e.target.files?.[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = () => {
      setPreviewImage(reader.result);
      setEnrollFeedback(null);
    };
    reader.readAsDataURL(file);
  }

  // Handle live snapshot capture from hardware camera
  async function handleCaptureFromCamera() {
    if (!connected) {
      setEnrollFeedback({ type: "error", text: "Camera service is offline. Please start the service first." });
      return;
    }
    setCapturingSnapshot(true);
    setEnrollFeedback(null);
    try {
      const res = await fetch(`${serverUrl}/api/snapshot?t=${Date.now()}`);
      if (!res.ok) throw new Error("Failed to capture snapshot from camera");
      const blob = await res.blob();
      const reader = new FileReader();
      reader.onload = () => {
        setPreviewImage(reader.result);
        setCapturingSnapshot(false);
      };
      reader.readAsDataURL(blob);
    } catch (err) {
      setCapturingSnapshot(false);
      setEnrollFeedback({ type: "error", text: err.message || "Capture failed" });
    }
  }

  // Enroll Face and Compute ArcFace Embedding
  async function handleEnrollPerson(e) {
    e.preventDefault();
    if (!personName.trim()) {
      setEnrollFeedback({ type: "error", text: "Please enter a name for the person." });
      return;
    }
    if (!previewImage) {
      setEnrollFeedback({ type: "error", text: "Please select or capture a face image first." });
      return;
    }

    setEnrollLoading(true);
    setEnrollFeedback(null);

    try {
      const res = await fetch(`${serverUrl}/api/register`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          name: personName.trim(),
          image: previewImage,
        }),
      });

      const data = await res.json();
      if (!res.ok || !data.success) {
        throw new Error(data.error || "Failed to process face embedding");
      }

      setEnrollFeedback({
        type: "success",
        text: `Success! Created 512-D ArcFace normalized embedding for "${personName}".`,
      });

      // Clear form
      setPersonName("");
      setPreviewImage(null);
      if (fileInputRef.current) fileInputRef.current.value = "";

      // Refresh list
      const facesRes = await fetch(`${serverUrl}/api/faces`);
      if (facesRes.ok) {
        const facesData = await facesRes.json();
        setFaces(facesData.faces || []);
      }
    } catch (err) {
      setEnrollFeedback({ type: "error", text: err.message || "Enrollment failed" });
    } finally {
      setEnrollLoading(false);
    }
  }

  // Delete Person
  async function handleDeletePerson(personId, name) {
    if (!confirm(`Are you sure you want to delete reference embeddings for "${name}"?`)) return;

    try {
      const res = await fetch(`${serverUrl}/api/faces/delete`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ id: personId }),
      });
      const data = await res.json();
      if (res.ok && data.success) {
        setFaces((prev) => prev.filter((p) => p.id !== personId));
      } else {
        alert(data.error || "Could not delete person");
      }
    } catch (err) {
      alert("Delete failed: " + err.message);
    }
  }

  return (
    <div className="face-tab-container">
      {/* Server Connection Bar with Tab-Specific Refresh Button */}
      <section className="connect panel">
        <div className="connect-label">
          <strong>Face Recognition Backend</strong>
          <span>
            {connected
              ? `Connected · Buffalo_L (${status?.model_status}) · ${faces.length} Registered Faces`
              : "Backend offline — run ./face_recognition_service/start_face_service.sh"}
          </span>
        </div>
        <input
          aria-label="Face Service URL"
          value={serverUrl}
          onChange={(e) => setServerUrl(e.target.value)}
          placeholder="http://localhost:5001"
        />
        <button
          className="refresh-btn"
          onClick={handleManualRefresh}
          disabled={refreshing}
          title="Reload only Face Recognition tab and test backend connection"
        >
          <span className={refreshing ? "spin-icon" : ""}>🔄</span>
          <span>{refreshing ? "Checking..." : "Refresh Tab"}</span>
        </button>
        <div className={`status ${connected ? "connected" : ""}`}>
          <span className="status-dot" />
          {connected ? "SERVICE ACTIVE" : "OFFLINE"}
        </div>
      </section>

      {/* Instant Notification Toast for Refresh Button */}
      {refreshFeedback && (
        <div className={`refresh-toast ${refreshFeedback.type}`}>
          <span>{refreshFeedback.text}</span>
          <button
            className="toast-close-btn"
            onClick={() => setRefreshFeedback(null)}
          >
            ✕
          </button>
        </div>
      )}

      {/* Overview Statistics Cards */}
      <section className="stats">
        <article className="panel stat">
          <span>Enrolled Identities</span>
          <strong>{faces.length} {faces.length === 1 ? "Person" : "Persons"}</strong>
        </article>
        <article className="panel stat">
          <span>Camera Stream</span>
          <strong>
            {cameraSource === "realsense" ? "RealSense D435i" : cameraSource === "laptop" ? "Laptop Webcam" : "Test Camera"}
          </strong>
        </article>
        <article className="panel stat">
          <span>Inference FPS</span>
          <strong>{status?.inference_fps ? `${status.inference_fps.toFixed(1)} FPS` : "—"}</strong>
        </article>
        <article className="panel stat">
          <span>3D Depth Spatial</span>
          <strong style={{ color: status?.show_depth ? "#78e3bd" : "#ff9c9c" }}>
            {status?.show_depth ? "XYZ Metric Active" : "Disabled"}
          </strong>
        </article>
      </section>

      {/* Main Grid: Video Stream + Enrollment & Gallery */}
      <div className="face-main-grid">
        {/* Left Column: Live Video Streams */}
        <div className="face-stream-column">
          <article className="panel">
            <div className="panel-title">
              <div>
                <h2>Live Hardware Face Recognition</h2>
                <p>Real-time ArcFace embedding match &amp; RealSense metric 3D deprojection</p>
              </div>
              <div className="pill-group">
                <button
                  className="secondary stream-reload-btn"
                  onClick={() => setStreamTimestamp(Date.now())}
                  title="Reconnect video stream without page reload"
                >
                  🔄 Stream
                </button>
                <span className="pill">{cameraSource.toUpperCase()}</span>
                <span className="pill">buffalo_l CPU</span>
              </div>
            </div>

            {/* Video Viewport */}
            <div className="face-video-viewport">
              {connected ? (
                <img
                  key={streamTimestamp}
                  className="face-video-stream"
                  src={`${serverUrl}/api/stream?t=${streamTimestamp}`}
                  alt="Live Face Recognition Feed"
                  onError={() => setConnected(false)}
                />
              ) : (
                <div className="empty-state face-offline-placeholder">
                  <div className="offline-card">
                    <h3>Face Recognition Backend Not Connected</h3>
                    <p>Start the Python InsightFace &amp; RealSense server to enable live detection and recognition:</p>
                    <pre className="code-box">cd face_recognition_service && ./start_face_service.sh</pre>
                    <button
                      className="secondary"
                      onClick={handleManualRefresh}
                      disabled={refreshing}
                      style={{ marginTop: "12px", padding: "9px 18px" }}
                    >
                      <span className={refreshing ? "spin-icon" : ""}>🔄</span>
                      <span>{refreshing ? "Checking Backend..." : "Check Backend Connection Now"}</span>
                    </button>
                    <p className="note" style={{ marginTop: "12px" }}>
                      Supports Intel RealSense D435i (Color + Aligned 3D Depth) and USB/laptop webcams.
                    </p>
                  </div>
                </div>
              )}
            </div>

            {/* Stream Settings & Source Bar */}
            <div className="face-stream-controls">
              <div className="control-field">
                <label>Camera Source:</label>
                <select
                  value={cameraSource}
                  onChange={(e) => handleCameraSourceChange(e.target.value)}
                  disabled={!connected}
                >
                  <option value="realsense">Intel RealSense D435i</option>
                  <option value="laptop">Laptop / USB Webcam</option>
                  <option value="test">Simulated Test Stream</option>
                </select>
              </div>

              <div className="control-field">
                <label>Cosine Match Threshold: <strong>{matchThreshold.toFixed(2)}</strong></label>
                <input
                  type="range"
                  min="0.10"
                  max="0.80"
                  step="0.05"
                  value={matchThreshold}
                  onChange={(e) => handleThresholdChange(e.target.value)}
                  disabled={!connected}
                />
              </div>

              <div className="control-field">
                <label>Inference Frame Skip: <strong>Every {frameSkip} frames</strong></label>
                <input
                  type="range"
                  min="1"
                  max="8"
                  step="1"
                  value={frameSkip}
                  onChange={(e) => handleFrameSkipChange(e.target.value)}
                  disabled={!connected}
                />
              </div>

              <div className="control-field toggle-field">
                <label>
                  <input
                    type="checkbox"
                    checked={showDepthView}
                    onChange={(e) => setShowDepthView(e.target.checked)}
                  />
                  Show RealSense Aligned Depth View
                </label>
              </div>
            </div>
          </article>

          {/* Optional Aligned Depth Colormap View */}
          {showDepthView && connected && cameraSource === "realsense" && (
            <article className="panel" style={{ marginTop: "14px" }}>
              <div className="panel-title">
                <div>
                  <h2>Intel RealSense Aligned Depth Colormap</h2>
                  <p>Colorized metric z16 depth stream aligned to color frame</p>
                </div>
                <span className="pill">640×480 @ 30fps</span>
              </div>
              <div className="face-video-viewport depth-viewport">
                <img
                  key={`depth-${streamTimestamp}`}
                  className="face-video-stream"
                  src={`${serverUrl}/api/depth_stream?t=${streamTimestamp}`}
                  alt="RealSense Colorized Depth Stream"
                />
              </div>
            </article>
          )}

          {/* Real-time Recognition Activity Log */}
          <article className="panel" style={{ marginTop: "14px" }}>
            <div className="panel-title">
              <div>
                <h2>Active Detections &amp; RealSense 3D Spatial Log</h2>
                <p>Live coordinates and identification confidence</p>
              </div>
              <button
                className="secondary"
                style={{ padding: "4px 8px", fontSize: "11px" }}
                onClick={() => setDetectionLogs([])}
              >
                Clear Log
              </button>
            </div>

            {detections.length > 0 ? (
              <div className="current-detections-banner">
                {detections.map((d, i) => (
                  <div key={i} className={`detection-badge ${d.matched ? "match" : "unknown"}`}>
                    <span className="badge-name">{d.name}</span>
                    <span className="badge-conf">{(d.confidence * 100).toFixed(1)}% match</span>
                    {d.position_3d && (
                      <span className="badge-coords">
                        XYZ=({d.position_3d.x.toFixed(2)}, {d.position_3d.y.toFixed(2)}, {d.position_3d.z.toFixed(2)})m
                      </span>
                    )}
                  </div>
                ))}
              </div>
            ) : (
              <div className="empty-subtext">No faces detected in current frame.</div>
            )}

            <div className="detection-log-list">
              {detectionLogs.map((log) => (
                <div key={log.id} className="log-row">
                  <span className="log-time">{log.time}</span>
                  <span className={`log-pill ${log.matched ? "match" : "unknown"}`}>{log.name}</span>
                  <span className="log-score">Similarity: {(log.confidence * 100).toFixed(1)}%</span>
                  {log.position_3d ? (
                    <span className="log-xyz">
                      XYZ: ({log.position_3d.x.toFixed(2)}, {log.position_3d.y.toFixed(2)}, {log.position_3d.z.toFixed(2)})m · Depth: {log.position_3d.depth.toFixed(2)}m
                    </span>
                  ) : (
                    <span className="log-xyz muted">2D Only</span>
                  )}
                </div>
              ))}
            </div>
          </article>
        </div>

        {/* Right Column: Person Enrollment & Saved Reference Database */}
        <div className="face-enroll-column">
          {/* Enroll New Face Form */}
          <article className="panel">
            <div className="panel-title">
              <div>
                <h2>Enroll Person &amp; Embeddings</h2>
                <p>Extract 512-D ArcFace normalized embedding</p>
              </div>
              <span className="pill">+ REGISTER</span>
            </div>

            <form onSubmit={handleEnrollPerson} className="enroll-form">
              <div className="form-group">
                <label>Person Full Name / Identifier</label>
                <input
                  type="text"
                  placeholder="e.g. Yash, Amrit, Dr. Sharma"
                  value={personName}
                  onChange={(e) => setPersonName(e.target.value)}
                  required
                />
              </div>

              {/* Source selection for reference face */}
              <div className="enroll-actions-row">
                <button
                  type="button"
                  className="secondary"
                  onClick={handleCaptureFromCamera}
                  disabled={!connected || capturingSnapshot}
                >
                  {capturingSnapshot ? "Capturing..." : "📷 Snapshot from Camera"}
                </button>
                <button
                  type="button"
                  className="secondary"
                  onClick={() => fileInputRef.current?.click()}
                >
                  📁 Browse Photo
                </button>
                <input
                  ref={fileInputRef}
                  type="file"
                  accept="image/jpeg,image/png,image/webp"
                  style={{ display: "none" }}
                  onChange={handleFileSelected}
                />
              </div>

              {/* Preview Box */}
              {previewImage && (
                <div className="preview-container">
                  <img src={previewImage} alt="Reference Preview" className="face-preview-img" />
                  <button
                    type="button"
                    className="preview-clear-btn"
                    onClick={() => {
                      setPreviewImage(null);
                      if (fileInputRef.current) fileInputRef.current.value = "";
                    }}
                  >
                    ✕
                  </button>
                </div>
              )}

              {enrollFeedback && (
                <div className={`enroll-feedback ${enrollFeedback.type}`}>
                  {enrollFeedback.text}
                </div>
              )}

              <button
                type="submit"
                className="enroll-submit-btn"
                disabled={enrollLoading || !personName.trim() || !previewImage || !connected}
              >
                {enrollLoading ? "Extracting ArcFace Embeddings..." : "Save Person & Embeddings"}
              </button>
            </form>
          </article>

          {/* Enrolled Persons Gallery */}
          <article className="panel" style={{ marginTop: "14px" }}>
            <div className="panel-title">
              <div>
                <h2>Enrolled Face Database</h2>
                <p>{faces.length} registered identities stored</p>
              </div>
              <span className="pill">{faces.length} PERSONS</span>
            </div>

            {faces.length === 0 ? (
              <div className="empty-state" style={{ height: "160px", position: "relative" }}>
                No persons registered yet. Use the form above to enroll a face!
              </div>
            ) : (
              <div className="faces-gallery">
                {faces.map((p) => (
                  <div key={p.id} className="face-card">
                    <img
                      src={`${serverUrl}/reference_faces/${p.image_file}`}
                      alt={p.name}
                      className="face-card-thumb"
                      onError={(e) => {
                        e.target.style.display = "none";
                      }}
                    />
                    <div className="face-card-details">
                      <strong>{p.name}</strong>
                      <span className="face-card-meta">512-D ArcFace Vector</span>
                      <span className="face-card-date">{p.created_at || "Enrolled"}</span>
                    </div>
                    <button
                      className="face-delete-btn"
                      title="Delete reference face"
                      onClick={() => handleDeletePerson(p.id, p.name)}
                    >
                      🗑
                    </button>
                  </div>
                ))}
              </div>
            )}
          </article>
        </div>
      </div>
    </div>
  );
}
