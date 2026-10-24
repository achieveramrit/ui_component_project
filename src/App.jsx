import React, { useEffect, useRef, useState } from "react";
import * as ROSLIB from "roslib";
import FaceRecognitionTab from "./components/FaceRecognitionTab.jsx";

const DEFAULT_ROS_URL = 'ws://192.168.1.50:9090'; // Replace your teammate's ROS laptop IP.

export default function App() {
  const [activeTab, setActiveTab] = useState('slam'); // 'slam' | 'face'

  // ROS & SLAM State
  const [rosUrl, setRosUrl] = useState(DEFAULT_ROS_URL);
  const [connected, setConnected] = useState(false);
  const [connectionMessage, setConnectionMessage] = useState('Not connected');
  const [mapMessage, setMapMessage] = useState('Waiting for /map…');
  const [mapMeta, setMapMeta] = useState(null);
  const [odom, setOdom] = useState({ x: 0, y: 0, yaw: 0, linear: 0, angular: 0 });
  const [rgbUrl, setRgbUrl] = useState('');
  const [depthUrl, setDepthUrl] = useState('');
  const [exploration, setExploration] = useState('Unknown');
  const rosRef = useRef(null);
  const topicsRef = useRef([]);
  const mapCanvasRef = useRef(null);

  function disconnect() {
    topicsRef.current.forEach((t) => { try { t.unsubscribe(); } catch {} });
    topicsRef.current = [];
    if (rosRef.current) {
      try { rosRef.current.close(); } catch {}
      rosRef.current = null;
    }
    setConnected(false);
    setConnectionMessage('Disconnected');
  }

  function connect() {
    disconnect();
    setConnectionMessage('Connecting…');
    const ros = new ROSLIB.Ros({ url: rosUrl.trim() });
    rosRef.current = ros;
    ros.on('connection', () => { setConnected(true); setConnectionMessage('Connected to ROS 2'); });
    ros.on('error', () => { setConnected(false); setConnectionMessage('Connection error — check IP, port, Wi-Fi and firewall'); });
    ros.on('close', () => { setConnected(false); setConnectionMessage('Connection closed'); });

    const subscribe = (name, messageType, callback) => {
      const topic = new ROSLIB.Topic({ ros, name, messageType, queue_length: 1 });
      topic.subscribe(callback);
      topicsRef.current.push(topic);
      return topic;
    };

    subscribe('/map', 'nav_msgs/OccupancyGrid', (msg) => {
      setMapMeta({ width: msg.info.width, height: msg.info.height, resolution: msg.info.resolution, frame: msg.header.frame_id || 'unknown' });
      setMapMessage('Receiving live occupancy grid');
      drawMap(msg);
    });

    subscribe('/odom', 'nav_msgs/Odometry', (msg) => {
      const p = msg.pose.pose.position;
      const q = msg.pose.pose.orientation;
      const yaw = Math.atan2(2 * (q.w * q.z + q.x * q.y), 1 - 2 * (q.y * q.y + q.z * q.z));
      setOdom({ x: p.x, y: p.y, yaw: yaw * 180 / Math.PI, linear: msg.twist.twist.linear.x, angular: msg.twist.twist.angular.z });
    });

    // Matches explore package: std_msgs/Bool on /explore/resume
    subscribe('/explore/resume', 'std_msgs/Bool', (msg) => setExploration(msg.data ? 'Enabled' : 'Paused'));

    // Compressed versions of camera streams
    subscribe('/camera/color/image_raw/compressed', 'sensor_msgs/CompressedImage', (msg) => setRgbUrl(`data:image/jpeg;base64,${msg.data}`));
    subscribe('/camera/depth/image_colorized/compressed', 'sensor_msgs/CompressedImage', (msg) => setDepthUrl(`data:image/jpeg;base64,${msg.data}`));
  }

  function drawMap(msg) {
    const canvas = mapCanvasRef.current;
    if (!canvas) return;
    const width = msg.info.width;
    const height = msg.info.height;
    if (!width || !height || msg.data.length !== width * height) return;
    canvas.width = width;
    canvas.height = height;
    const ctx = canvas.getContext('2d');
    const pixels = ctx.createImageData(width, height);
    for (let i = 0; i < msg.data.length; i++) {
      const value = msg.data[i];
      const shade = value < 0 ? 105 : Math.round(255 - Math.max(0, Math.min(100, value)) * 2.35);
      const j = i * 4;
      pixels.data[j] = shade;
      pixels.data[j + 1] = shade;
      pixels.data[j + 2] = shade;
      pixels.data[j + 3] = 255;
    }
    const rowBytes = width * 4;
    const temp = new Uint8ClampedArray(rowBytes);
    for (let y = 0; y < Math.floor(height / 2); y++) {
      const top = y * rowBytes;
      const bottom = (height - 1 - y) * rowBytes;
      temp.set(pixels.data.subarray(top, top + rowBytes));
      pixels.data.copyWithin(top, bottom, bottom + rowBytes);
      pixels.data.set(temp, bottom);
    }
    ctx.putImageData(pixels, 0, 0);
  }

  function publishVelocity(linear, angular) {
    if (!connected || !rosRef.current) return;
    const cmd = new ROSLIB.Topic({ ros: rosRef.current, name: '/cmd_vel', messageType: 'geometry_msgs/Twist' });
    cmd.publish({ linear: { x: linear, y: 0, z: 0 }, angular: { x: 0, y: 0, z: angular } });
  }

  function setExplorationEnabled(enabled) {
    if (!connected || !rosRef.current) return;
    const topic = new ROSLIB.Topic({ ros: rosRef.current, name: '/explore/resume', messageType: 'std_msgs/Bool' });
    topic.publish({ data: enabled });
    setExploration(enabled ? 'Start requested' : 'Pause requested');
  }

  function stopRobot() {
    publishVelocity(0, 0);
    setExplorationEnabled(false);
  }

  useEffect(() => () => disconnect(), []);

  return (
    <div className="app-shell">
      {/* Top Header */}
      <header className="header">
        <div>
          <div className="eyebrow">AUTONOMOUS GROUND VEHICLE · HARDWARE SYSTEM</div>
          <h1>SLAM Explorer &amp; Perception Hub</h1>
          <p className="subheading">Live mapping · RealSense perception · ArcFace facial recognition · Frontier exploration</p>
        </div>
        <div className={`status ${connected ? 'connected' : ''}`}>
          <span className="status-dot" />
          {connected ? 'ROS 2 CONNECTED' : 'ROS OFFLINE'}
        </div>
      </header>

      {/* Primary Navigation Tabs */}
      <nav className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'slam' ? 'active' : ''}`}
          onClick={() => setActiveTab('slam')}
        >
          <span className="tab-icon">🗺️</span>
          <span>Robot SLAM &amp; Navigation</span>
        </button>
        <button
          className={`tab-btn ${activeTab === 'face' ? 'active' : ''}`}
          onClick={() => setActiveTab('face')}
        >
          <span className="tab-icon">👤</span>
          <span>Face Recognition (InsightFace &amp; RealSense)</span>
        </button>
      </nav>

      {/* Tab 1: Robot SLAM & Navigation Dashboard */}
      {activeTab === 'slam' && (
        <div className="tab-content">
          <section className="connect panel">
            <div className="connect-label">
              <strong>ROS 2 WebSocket</strong>
              <span>{connectionMessage}</span>
            </div>
            <input
              aria-label="ROS WebSocket URL"
              value={rosUrl}
              onChange={(e) => setRosUrl(e.target.value)}
              placeholder="ws://ROBOT_IP:9090"
            />
            <button onClick={connect} disabled={!rosUrl.trim()}>Connect</button>
            <button className="secondary" onClick={disconnect}>Disconnect</button>
          </section>

          <section className="stats">
            <Stat label="Position X" value={`${odom.x.toFixed(2)} m`} />
            <Stat label="Position Y" value={`${odom.y.toFixed(2)} m`} />
            <Stat label="Heading" value={`${odom.yaw.toFixed(1)}°`} />
            <Stat label="Linear speed" value={`${odom.linear.toFixed(2)} m/s`} />
          </section>

          <section className="main-grid">
            <article className="panel map-panel">
              <div className="panel-title">
                <div>
                  <h2>Live SLAM Map</h2>
                  <p>{mapMessage}</p>
                </div>
                <span className="pill">/map</span>
              </div>
              <div className="map-frame">
                <canvas ref={mapCanvasRef} />
                {!mapMeta && <div className="empty-state">Connect to ROS 2 and wait for the SLAM map.</div>}
              </div>
              <div className="map-footer">
                {mapMeta
                  ? `${mapMeta.width} × ${mapMeta.height} cells · ${mapMeta.resolution.toFixed(3)} m/cell · frame: ${mapMeta.frame}`
                  : 'Waiting for occupancy grid'}
              </div>
            </article>

            <div className="camera-stack">
              <CameraPanel
                title="RGB Camera"
                topic="/camera/color/image_raw/compressed"
                image={rgbUrl}
                empty="No compressed RGB stream received"
              />
              <CameraPanel
                title="Colorized Depth"
                topic="/camera/depth/image_colorized/compressed"
                image={depthUrl}
                empty="Ask teammate to enable colorized depth + compression"
              />
            </div>
          </section>

          <section className="control-grid">
            <article className="panel">
              <div className="panel-title">
                <div>
                  <h2>Autonomous Exploration</h2>
                  <p>Topic: /explore/resume · std_msgs/Bool</p>
                </div>
                <span className="pill">{exploration}</span>
              </div>
              <div className="button-row">
                <button onClick={() => setExplorationEnabled(true)} disabled={!connected}>
                  Start / Resume
                </button>
                <button className="secondary" onClick={() => setExplorationEnabled(false)} disabled={!connected}>
                  Pause Exploration
                </button>
              </div>
              <p className="note">
                This matches the explore package in your repository. Start/resume and pause do not guarantee an immediate physical stop.
              </p>
            </article>

            <article className="panel">
              <div className="panel-title">
                <div>
                  <h2>Manual Motion</h2>
                  <p>Topic: /cmd_vel · geometry_msgs/Twist</p>
                </div>
                <span className="pill warning">Test safely</span>
              </div>
              <div className="motion-pad">
                <button disabled={!connected} onClick={() => publishVelocity(0.15, 0)}>Forward</button>
                <div>
                  <button disabled={!connected} onClick={() => publishVelocity(0, 0.35)}>Turn left</button>
                  <button className="stop-button" disabled={!connected} onClick={() => publishVelocity(0, 0)}>Zero velocity</button>
                  <button disabled={!connected} onClick={() => publishVelocity(0, -0.35)}>Turn right</button>
                </div>
                <button disabled={!connected} onClick={() => publishVelocity(-0.12, 0)}>Reverse</button>
              </div>
              <p className="note">
                Use only with the robot lifted or in a clear test area. Verify your motor controller's stop behavior before moving the real robot.
              </p>
            </article>
          </section>
        </div>
      )}

      {/* Tab 2: Dedicated Face Recognition Hub */}
      {activeTab === 'face' && (
        <div className="tab-content">
          <FaceRecognitionTab />
        </div>
      )}

      <footer>
        Autonomous Ground Vehicle Dashboard · ROS 2 Humble &amp; Intel RealSense D435i Deep Perception
      </footer>
    </div>
  );
}

function Stat({ label, value }) {
  return (
    <article className="panel stat">
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function CameraPanel({ title, topic, image, empty }) {
  return (
    <article className="panel camera-panel">
      <div className="panel-title">
        <div>
          <h2>{title}</h2>
          <p>{topic}</p>
        </div>
        <span className="pill">CAMERA</span>
      </div>
      {image ? <img src={image} alt={title} /> : <div className="camera-empty">{empty}</div>}
    </article>
  );
}
