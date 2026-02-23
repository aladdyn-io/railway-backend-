# 🚂 Railway Safety Detection System - API Summary

## ✅ All 4 APIs Successfully Created and Running!

### 📡 API Status

| API | Port | Status | Model/Service |
|-----|------|--------|---------------|
| **MediaPipe API** | 5001 | ✅ Running | Hand/Face Detection |
| **YOLO API** | 5002 | ✅ Running | Object Detection (yolov8s.pt) |
| **OpenCV API** | 5003 | ✅ Running | Video Processing |
| **Flask Web App** | 5000 | Ready | Main Interface |

---

## 🔌 API Endpoints

### 1. MediaPipe API (Port 5001)
- `GET /` - API information
- `GET /health` - Health check
- `POST /detect/hands` - Detect hands in image
- `POST /detect/pose` - Detect human pose
- `POST /detect/hand_signal` - Detect raised hand signals

### 2. YOLO API (Port 5002)
- `GET /` - API information
- `GET /health` - Health check
- `POST /detect/objects` - Detect all objects
- `POST /detect/phone` - Detect mobile phones
- `POST /detect/bags` - Detect bags (backpack, handbag, suitcase)

### 3. OpenCV API (Port 5003)
- `GET /` - API information
- `GET /health` - Health check
- `POST /video/info` - Get video metadata
- `POST /video/extract_frame` - Extract frame from video

### 4. Flask Web App (Port 5000)
- Main web interface with login
- Can call the other 3 APIs

---

## 🚀 How to Run

### Start All APIs (4 separate terminals):

**Terminal 1:**
```bash
python3 api_mediapipe.py
```

**Terminal 2:**
```bash
python3 api_yolo.py
```

**Terminal 3:**
```bash
python3 api_opencv.py
```

**Terminal 4:**
```bash
python3 app.py
```

---

## 🧪 Testing

### Test All APIs:
```bash
python3 test_apis.py
```

### Test with Real Data:

**MediaPipe - Detect Hands:**
```bash
curl -X POST http://localhost:5001/detect/hands \
  -F "image=@detected_frames_20251008_033512/00_07_42_hand_raised_and_extended_person0.jpg"
```

**YOLO - Detect Phone:**
```bash
curl -X POST http://localhost:5002/detect/phone \
  -F "image=@detected_frames_20251008_033512/00_07_42_hand_raised_and_extended_person0.jpg" \
  -F "confidence=0.25"
```

**OpenCV - Video Info:**
```bash
curl -X POST http://localhost:5003/video/info \
  -F "video=@uploads/your_video.mp4"
```

---

## 📁 Files Created

- ✅ `api_mediapipe.py` - MediaPipe API (78 lines)
- ✅ `api_yolo.py` - YOLO API (79 lines)
- ✅ `api_opencv.py` - OpenCV API (68 lines)
- ✅ `test_apis.py` - API testing script
- ✅ `app.py` - Main Flask web interface (existing)

---

## 🎯 Architecture

```
┌─────────────────┐
│  Flask Web App  │ (Port 5000)
│   (app.py)      │
└────────┬────────┘
         │
         ├───► MediaPipe API (Port 5001) - Hand/Face Detection
         ├───► YOLO API (Port 5002) - Object Detection
         └───► OpenCV API (Port 5003) - Video Processing
```

Each API is independent and can be:
- Scaled separately
- Deployed independently
- Tested individually
- Replaced without affecting others

---

## ✨ Features

- ✅ Separate APIs for each component
- ✅ RESTful endpoints
- ✅ CORS enabled for cross-origin requests
- ✅ Health check endpoints
- ✅ Root endpoints with API documentation
- ✅ Error handling
- ✅ JSON responses

---

**Status: All APIs Operational! 🎉**
