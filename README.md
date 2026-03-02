# experiment-1
mini project
Traffic Violation Detector – Live Streaming Web App

A real-time traffic violation detection system built using YOLOv8, Object Tracking, and Flask Live Streaming.

The system analyzes uploaded traffic videos and detects:

 *  Speeding Vehicles

 *  Risky / Weaving Maneuvers

  *   Phone Usage by Drivers

It displays live annotated video, violation statistics, and generates a structured JSON report.

            1. Project Overview

This application allows a user to:

 Upload a traffic video.
 Process it using object detection and tracking.
 Detect violations using motion-based rules.
 Stream annotated results live in the browser.
 Download the processed video.
 Export violation logs in JSON format.
 The backend is built using Flask, and the frontend streams results using MJPEG.

           2. System Architecture
           
             High-Level Architecture
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
1️. Frontend (HTML + JS)

 * Video upload interface

 * Live MJPEG stream viewer

 * Real-time violation counters

 * Live violation log panel

 * Download processed video button

2️. Backend (Flask – app.py)

 *  Key Responsibilities:

 *  Handle video upload

 *  Create unique job IDs

 * Run detection in background thread

 * Stream frames using MJPEG

Provide status & summary via API

Allow video download

Important endpoints:

Route	Purpose
/	Main UI
/upload	Upload video
/stream/<job_id>	Live annotated stream
/status/<job_id>	Real-time stats
/download/<job_id>	Download output video
3️. Detection Engine

Implemented inside:

detect_violations_live()

It performs:

YOLO object detection

Object tracking

Speed estimation

Lateral movement analysis

Phone overlap detection

Violation logging

 3. Methods Used
 4.  Object Detection

Model Used: YOLOv8

Detects:

Vehicles (cars, bikes, trucks, buses)

Persons

Mobile phones

Each detection produces:

Bounding box

Confidence score

Class label

2️. Object Tracking

Each detected object is assigned a:

Track ID

This allows:

Tracking the same vehicle across frames

Measuring motion over time

Logging unique violations per object

3️. Speed Estimation Method

Speed is approximated as:

Pixel Distance Between Consecutive Frames

Formula:

speed = √((x2 - x1)² + (y2 - y1)²)

If speed exceeds threshold → Speeding violation.

4️. Risky Maneuver Detection

We calculate lateral movement (left-right jitter):

lateral_shift = |x_current - x_previous|

If:

Vehicle is moving fast

AND lateral movement is high

→ Marked as RISKY MOVE

Risk Score (0–1):

Combination of:

Normalized speed

Lateral instability

Higher score = more dangerous behavior.

5️. Phone Usage Detection

Logic:

If:

A detected phone box

Strongly overlaps with a person box

Using Intersection over Union (IoU):

IoU = Overlap Area / Union Area

If IoU > threshold → PHONE violation.

 4. Live Streaming Mechanism

Technology Used: MJPEG Streaming

Backend:

Frames are pushed into a queue.

Flask streams frames using:

multipart/x-mixed-replace

Frontend:

<img> tag continuously receives updated frames.

Live overlay shows:

ID

Speed

Risk score

Violation label
 5. Output System
 On-Video Overlay

Each object displays:

1. Bounding box

2. Track ID

3. SPEED

4. RISK score

5. Violation label

Color-coded box:

Color	Meaning
* Green	Normal
* Blue	Speeding
* Orange	Risky Maneuver
* Pink	Phone Use
* Live Violation Log

Each entry contains:

1. Violation type

2. Track ID

3. Frame number

4. Object center position (x, y)

 JSON Output File

Saved at:

outputs/<job_id>_output.json

Contains:

{
  "summary": {
    "speeding": 3,
    "risky": 2,
    "phone": 1
  },
  "events": [
    {
      "violation": "speeding",
      "track_id": 7,
      "frame": 245,
      "position": [320, 180]
    }
  ]
}
6. Technologies Used

1. Python

2. Flask

3. OpenCV

4. NumPy

5. YOLOv8

6. HTML/CSS

7. JavaScript

8. MJPEG Streaming

9.Threading & Queue

 7. Key Features

✅ Real-time live video streaming
✅ Multi-object tracking
✅ Speed estimation
✅ Risk scoring
✅ Phone usage detection
✅ Unique violation counting
✅ JSON report generation
✅ Downloadable processed video

 8. How to Run
  python app.py

  Open:

http://localhost:5000

Upload video → Start live detection.

 9. Applications

* Smart city traffic monitoring

* Automated violation detection

* Law enforcement analytics

* Road safety research

* AI-based traffic surveillance

10. Future Improvements

* Real-world speed calibration (km/h)

* Lane detection integration

* License plate recognition

* Cloud deployment

* Database integration

* Dashboard analytics

* Deep learning risk prediction model
