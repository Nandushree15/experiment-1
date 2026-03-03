Traffic Violation Detector – Live Streaming Web App

A real-time traffic violation detection system built using YOLOv8, Object Tracking, and Flask Live Streaming.

* The system analyzes uploaded traffic videos and detects:

* Speeding Vehicles

* Risky / Weaving Maneuvers

* Phone Usage by Drivers

It displays live annotated video, violation statistics, and generates a structured JSON report.
1. Project Overview

This application allows users to:

1. Upload a traffic video

2. Process it using object detection and tracking

3. Detect violations using motion-based rules

4. Stream annotated results live in the browser

5. Download the processed video

6. Export violation logs in JSON format

Backend: Flask
Frontend: MJPEG-based live streaming

2. System Architecture

User Upload
     ↓
Flask Server
     ↓
YOLOv8 Detection
     ↓
Object Tracking (Track IDs)
     ↓
Motion Analysis (Speed + Weaving)
     ↓
Rule-Based Violation Engine
     ↓
Live MJPEG Stream + JSON Log

Component Architecture
1. Frontend (HTML + JavaScript)

* Video upload interface

* Live MJPEG stream viewer

* Real-time violation counters

* Live violation log panel

* Download processed video button
  
2. Backend (Flask – app.py)

Key Responsibilities:

1. Handle video upload

2. Create unique job IDs

3. Run detection in background thread

4. Stream frames using MJPEG

5. Provide status and summary via API

6. Allow video download

3. Detection Engine
Implemented inside:

CODE
detect_violations_live()

Performs:

* YOLO object detection

* Object tracking

* Speed estimation

* Lateral movement analysis

* Phone overlap detection

* Violation logging

3. Methods Used
1. Object Detection

Model Used: YOLOv8

Detects:

1.Vehicles (cars, bikes, trucks, buses)

2.Persons

3.Mobile phones

Each detection provides:

1.Bounding box

2.Confidence score

3.Class label

2. Object Tracking
Each detected object is assigned a Track ID.

This allows:

* Tracking the same vehicle across frames

* Measuring motion over time

* Logging unique violations per object
  
3. Speed Estimation Method
Speed is approximated as:

Pixel Distance Between Consecutive Frames
speed = √((x2 - x1)² + (y2 - y1)²)

4. Risky Maneuver Detection
Lateral movement is calculated as:

lateral_shift = |x_current - x_previous|

5. Output System
On-Video Overlay

Each object displays:

1.Bounding box
2.Track ID
3.Speed
4.Risk score
5.Violation label

| Color  | Meaning        |
| ------ | -------------- |
| Green  | Normal         |
| Blue   | Speeding       |
| Orange | Risky Maneuver |
| Pink   | Phone Use      |


6. Technologies Used

* Python
* Flask
* OpenCV
* NumPy
* YOLOv8
* HTML
* CSS
* JavaScript
* MJPEG Streaming

