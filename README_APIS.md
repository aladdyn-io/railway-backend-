# Pilot Eye Analytics Hub - API Documentation

## Overview

This project consists of **4 separate APIs** that work together to provide video analytics for railway driver safety monitoring:

1. **MediaPipe API** (Port 5001) - Hand, Face, and Pose Detection
2. **YOLO API** (Port 5002) - Object, Phone, and Bag Detection  
3. **OpenCV API** (Port 5003) - Video Processing
4. **Flask Web API** (Port 5000) - Main Interface

## Quick Start

### 1. Install Dependencies

```bash
pip install flask flask-cors opencv-python numpy mediapipe ultralytics requests
```

### 2. Start All APIs

```bash
# Make scripts executable (first time only)
chmod +x start_apis.sh stop_apis.sh

# Start all APIs
./start_apis.sh
```

### 3. Start Frontend

```bash
cd pilot-eye-analytics-hub
npm install
npm run dev
```

The frontend will be available at `http://localhost:8082` (or the port Vite assigns)

## API Details

### 1. MediaPipe API (Port 5001)

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /detect/hands` - Detect hand landmarks
- `POST /detect/pose` - Detect pose landmarks
- `POST /detect/hand_signal` - Detect raised hand signals
- `POST /detect/face` - Detect face landmarks
- `POST /detect/drowsiness` - Detect drowsiness/microsleep

**Example:**
```bash
curl -X POST http://localhost:5001/detect/hands \
  -F "image=@test_image.jpg"
```

### 2. YOLO API (Port 5002)

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /detect/objects` - Detect all objects
- `POST /detect/phone` - Detect phones
- `POST /detect/bags` - Detect bags/backpacks

**Example:**
```bash
curl -X POST http://localhost:5002/detect/phone \
  -F "image=@test_image.jpg" \
  -F "confidence=0.25"
```

### 3. OpenCV API (Port 5003)

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `POST /video/info` - Get video information
- `POST /video/extract_frame` - Extract single frame
- `POST /video/extract_frames` - Extract multiple frames

**Example:**
```bash
curl -X POST http://localhost:5003/video/info \
  -F "video=@test_video.mp4"
```

### 4. Flask Web API (Port 5000)

**Endpoints:**
- `GET /` - API information
- `GET /health` - Health check
- `GET /api/status` - Check status of all APIs

**Example:**
```bash
curl http://localhost:5000/api/status
```

## Project Structure

```
Railways Project/
├── api_mediapipe.py      # MediaPipe API
├── api_yolo.py           # YOLO API
├── api_opencv.py         # OpenCV API
├── api_flask.py          # Flask Web API
├── start_apis.sh         # Start all APIs
├── stop_apis.sh          # Stop all APIs
├── logs/                 # API logs
├── uploads/              # Temporary video uploads
├── output/               # Processed outputs
└── pilot-eye-analytics-hub/  # Frontend React app
    └── src/
        └── services/
            └── api.ts   # Frontend API client
```

## Features

### Detection Capabilities

1. **Hand Detection** - Detects hand landmarks and gestures
2. **Pose Detection** - Detects body pose landmarks
3. **Hand Signals** - Detects raised hand signals
4. **Face Detection** - Detects faces and landmarks
5. **Drowsiness Detection** - Detects microsleep/drowsiness using eye aspect ratio
6. **Object Detection** - Detects all objects using YOLO
7. **Phone Detection** - Specifically detects mobile phones
8. **Bag Detection** - Detects bags, backpacks, suitcases

### Video Processing

- Extract frames from videos
- Get video metadata (duration, fps, resolution)
- Process videos frame-by-frame for detection

## Troubleshooting

### APIs not starting

1. Check if ports are already in use:
```bash
lsof -i :5000
lsof -i :5001
lsof -i :5002
lsof -i :5003
```

2. Kill existing processes:
```bash
./stop_apis.sh
```

3. Check logs:
```bash
tail -f logs/mediapipe.log
tail -f logs/yolo.log
tail -f logs/opencv.log
tail -f logs/flask.log
```

### YOLO model not loading

The YOLO model (`yolov8s.pt`) will be automatically downloaded on first use. Make sure you have internet connection.

### MediaPipe timestamp errors

MediaPipe APIs create new instances for each request to avoid timestamp mismatch errors. If you still see errors, restart the MediaPipe API.

### Frontend not connecting

1. Make sure all APIs are running: `./start_apis.sh`
2. Check API health: `curl http://localhost:5001/health`
3. Check browser console for CORS errors
4. Verify API URLs in `pilot-eye-analytics-hub/src/services/api.ts`

## Development

### Running APIs Individually

```bash
# MediaPipe API
python3 api_mediapipe.py

# YOLO API
python3 api_yolo.py

# OpenCV API
python3 api_opencv.py

# Flask API
python3 api_flask.py
```

### Testing APIs

```bash
# Test MediaPipe
curl http://localhost:5001/health

# Test YOLO
curl http://localhost:5002/health

# Test OpenCV
curl http://localhost:5003/health

# Test Flask
curl http://localhost:5000/api/status
```

## License

This project is for railway driver safety monitoring.



