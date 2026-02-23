#!/bin/bash

# Stop all 4 APIs

echo "Stopping all APIs..."

lsof -ti:5000 | xargs kill -9 2>/dev/null && echo "✅ Stopped Flask API (5000)" || echo "⚠️  Flask API (5000) not running"
lsof -ti:5001 | xargs kill -9 2>/dev/null && echo "✅ Stopped MediaPipe API (5001)" || echo "⚠️  MediaPipe API (5001) not running"
lsof -ti:5002 | xargs kill -9 2>/dev/null && echo "✅ Stopped YOLO API (5002)" || echo "⚠️  YOLO API (5002) not running"
lsof -ti:5003 | xargs kill -9 2>/dev/null && echo "✅ Stopped OpenCV API (5003)" || echo "⚠️  OpenCV API (5003) not running"

echo ""
echo "All APIs stopped!"



