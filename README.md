# SLAM Explorer Dashboard (Windows frontend)

This frontend is tailored to the supplied Bot AGV repository. The repository README identifies ROS 2 Humble and these interfaces:

- `/map` — `nav_msgs/OccupancyGrid`
- `/odom` — `nav_msgs/Odometry` (if the running launch publishes it)
- `/explore/resume` — `std_msgs/Bool` (`true` resume, `false` pause)
- `/cmd_vel` — `geometry_msgs/Twist`
- `/camera/color/image_raw` — `sensor_msgs/Image` (raw)
- `/camera/depth/image_rect_raw` — `sensor_msgs/Image` (raw 16UC1)
- Optional `/camera/depth/image_colorized` — `sensor_msgs/Image` (BGR8, disabled by default in the custom driver)

## Run on Windows

1. Install Node.js LTS and VS Code.
2. Extract this folder, open it in VS Code.
3. In VS Code Terminal run `npm install`.
4. Run `npm run dev` and open the localhost URL Vite prints.
5. In the UI, replace `ws://192.168.1.50:9090` with the ROS computer's reachable IP and click Connect.

## ROS computer / teammate checklist

1. Start the existing ROS 2 system and confirm `/map`, `/odom`, `/explore/resume`, and `/cmd_vel` with `ros2 topic list` and `ros2 topic info <topic>`.
2. Install rosbridge for the same ROS distribution and run `ros2 launch rosbridge_server rosbridge_websocket_launch.xml`.
3. Ensure Windows can reach `ws://ROBOT_IP:9090`; allow port 9090 through the Ubuntu firewall only on a trusted network.
4. Raw ROS `sensor_msgs/Image` topics are not directly rendered by this starter. The camera cards expect `sensor_msgs/CompressedImage` at `/camera/color/image_raw/compressed` and `/camera/depth/image_colorized/compressed`.
5. The custom RealSense driver in the supplied repo publishes raw RGB and raw 16-bit depth. It sets `publish_colorized_depth = False` by default. Teammate must enable/publish colorized depth and configure image transport/compression (or adapt the UI to the exact image transport in use) before the camera cards show images.

## Safety

Manual controls publish real `/cmd_vel` commands. Test with wheels lifted or a safe clear space, use low speeds, and ensure the hardware/controller has a reliable emergency stop. A zero Twist is not a substitute for a hardware emergency stop and does not necessarily cancel a Nav2 goal.

This is a frontend starter, not a replacement for ROS, SLAM, Nav2, or the exploration node.
