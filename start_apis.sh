#!/bin/bash

# Start all 4 APIs for Pilot Eye Analytics Hub
# MediaPipe API (5001), YOLO API (5002), OpenCV API (5003), Flask API (5000)

echo "🚀 Starting all APIs..."
echo ""

# Kill any existing processes on these ports
echo "Cleaning up existing processes..."
lsof -ti:5000 | xargs kill -9 2>/dev/null
lsof -ti:5001 | xargs kill -9 2>/dev/null
lsof -ti:5002 | xargs kill -9 2>/dev/null
lsof -ti:5003 | xargs kill -9 2>/dev/null
sleep 2

# Start MediaPipe API (port 5001)
echo "Starting MediaPipe API on port 5001..."
cd "$(dirname "$0")"
python3 api_mediapipe.py > logs/mediapipe.log 2>&1 &
MEDIAPIPE_PID=$!
echo "MediaPipe API started (PID: $MEDIAPIPE_PID)"

# Start YOLO API (port 5002)
echo "Starting YOLO API on port 5002..."
python3 api_yolo.py > logs/yolo.log 2>&1 &
YOLO_PID=$!
echo "YOLO API started (PID: $YOLO_PID)"

# Start OpenCV API (port 5003)
echo "Starting OpenCV API on port 5003..."
python3 api_opencv.py > logs/opencv.log 2>&1 &
OPENCV_PID=$!
echo "OpenCV API started (PID: $OPENCV_PID)"

# Start Flask API (port 5000)
echo "Starting Flask API on port 5000..."
python3 api_flask.py > logs/flask.log 2>&1 &
FLASK_PID=$!
echo "Flask API started (PID: $FLASK_PID)"

# Wait a bit for APIs to start
sleep 5

# Check if APIs are running
echo ""
echo "Checking API health..."
curl -s http://localhost:5001/health > /dev/null && echo "✅ MediaPipe API (5001) - OK" || echo "❌ MediaPipe API (5001) - FAILED"
curl -s http://localhost:5002/health > /dev/null && echo "✅ YOLO API (5002) - OK" || echo "❌ YOLO API (5002) - FAILED"
curl -s http://localhost:5003/health > /dev/null && echo "✅ OpenCV API (5003) - OK" || echo "❌ OpenCV API (5003) - FAILED"
curl -s http://localhost:5000/health > /dev/null && echo "✅ Flask API (5000) - OK" || echo "❌ Flask API (5000) - FAILED"

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "All APIs started!"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "MediaPipe API: http://localhost:5001"
echo "YOLO API:      http://localhost:5002"
echo "OpenCV API:    http://localhost:5003"
echo "Flask API:     http://localhost:5000"
echo ""
echo "Logs are in the 'logs' directory"
echo "To stop all APIs, run: ./stop_apis.sh"
echo ""



