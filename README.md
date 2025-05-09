# OREI-Cam Real-Time Vision and Precision Spraying System

## Overview

This ROS 2 workspace implements a real-time vision and precision spraying system designed for agroecological applications. The system is deployed on a mobile robot equipped with an NVIDIA AGX Orin, a camera system, LiDAR, GPS, and a 25-nozzle sprayer boom.

The primary goal is to detect plants (crops and weeds) in real-world field conditions, track them as the robot moves, and then precisely actuate sprayer nozzles to target specific objects (e.g., weeds) based on configurable strategies, while minimizing off-target application.

The system is modular, with key functionalities encapsulated in dedicated ROS 2 packages:

1.  **`vision_system`**: Performs image acquisition, preprocessing, motion estimation (frame-to-frame displacement), YOLOv11-based object detection, PSEZ-crop matching, and object tracking.
2.  **`sprayer_system`**: Manages detected objects, applies spraying strategies, predicts object ground positions, schedules nozzle activations considering latencies, and interfaces with the sprayer hardware.
3.  **Camera & Sensor Packages**:
    *   `camera_node_front`: Interfaces with the physical camera, handles triggering and image metadata.
    *   `camera_trigger`: Generates camera trigger signals based on GPS movement.
    *   `camera_height`: Estimates camera height above ground using LiDAR data.
    *   `image_writer`: Persists images and associated metadata to disk.
    *   `septentrio_gnss_driver`: Interfaces with the Septentrio GNSS/GPS unit.
    *   `lidar_driver`: Basic driver for LiDAR distance readings.
4.  **Hardware Interface & Control Packages**:
    *   `serial_driver`: Provides a generic serial communication bridge.
    *   `relay_driver`: Manages the serial commands to the 25-nozzle relay board.
    *   `spraying_state_control`: (Potentially) Manages overall spraying state based on geofenced zones (shapefiles).
5.  **Utility & Interface Packages**:
    *   `orei_cam_interfaces`: Defines custom ROS messages for image data with rich metadata.
    *   `vision_interfaces`: Defines custom ROS messages for tracking results.
    *   `orei_cam_gui` (Optional): A simple GUI for displaying camera feed and system status.

## Prerequisites

*   **ROS 2**: Humble
*   **Python**: 3.9
*   **NVIDIA JetPack**: 6.0

## Installation

To be described.

## Running the system

To be described.

## Configuration

The system is highly configurable via YAML files located in the `config` directory of the respective packages (`vision_system/config`, `sprayer_system/config`, etc.).

*   **`vision_system/config/vision_system_config.yaml`**: Configures image processing, motion estimation, inference engine (model path, thresholds), model metadata (class names), and object tracker parameters.
*   **`sprayer_system/config/sprayer_config.yaml`**: Configures ROS topics, control loop frequency, camera parameters (for coordinate transforms), motion model settings, spraying strategies, nozzle layout (calibration file or default), timing latencies, and hardware interface details.
*   **`sprayer_system/config/nozzle_calibration.yaml`**: Contains nozzle spray coverage areas as polygons defined in image pixel coordinates.
*   **`septentrio_gnss_driver/config/rover_node.yaml`**: Configuration for the GPS/GNSS receiver.
*   **`serial_driver/params/relay_board.params.yaml`**: Configuration for the serial port connected to the relay board.