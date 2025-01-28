# Hand Tracking Project

## Overview

This project performs real-time hand detection and visualization using OpenCV and MediaPipe. It detects and tracks hand landmarks from a webcam feed, visualizes key points and connections, and overlays dynamic information such as runtime, FPS, and other metrics.

## Features

- **Real-Time Hand Tracking**: Utilizes MediaPipe to detect and track multiple hands.
- **Dynamic Visual Overlays**: Displays information overlays including runtime, FPS, number of hands detected, and more.
- **Customizable Visual Elements**: Includes classes for drawing dynamic cards and anchored text labels.
- **Vignette Effect**: Enhances visual clarity by applying a semi-transparent vignette overlay.
- **Modular Architecture**: Organized into separate modules for components, utilities, and testing to ensure maintainability and scalability.
- **Logging**: Implements logging for better debugging and monitoring of application state.

## Setup Instructions

1. **Clone the Repository**

   ```bash
   git clone https://github.com/yourusername/hand_detection_project.git
   cd hand_detection_project
   ```

2. **Install Dependencies**

   ```bash
   pip install -r requirements.txt
   ```

## Usage Guidelines

- **Webcam Index**: By default, the application attempts to use camera_index=1. If this fails, it falls back to the default camera (camera_index=0). You can modify the camera_index in main.py if needed.

- **Visual Overlays**:
  - **Details Card**: Displays runtime, FPS, number of hands detected, and hand confidence scores.
  - **Circle Visualization**: Draws a dynamic circle between the thumb tip and index finger tip of the right hand, displaying its radius.
  - **Text Labels**: Anchors labels to specific landmarks, such as "Pointer 556" on the left middle finger.

- **Logging**: The application logs important events, warnings, and errors to the console for easier debugging and monitoring.
