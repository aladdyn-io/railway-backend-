# CVVRS Backend API

Video processing API for Pilot Eye Analytics Hub - Railway deployment.

## Features

- **Phone Detection**: YOLO-based phone usage detection
- **Hand Signal Detection**: MediaPipe-based hand signal recognition
- **Station Alert Compliance**: GPS and timing-based station alert validation
- **Microsleep Detection**: Eye aspect ratio analysis for drowsiness
- **Bag Detection**: Object detection for bags and luggage
- **General Object Detection**: YOLO object detection

## Requirements

- Python 3.9+
- See `requirements.txt` for dependencies

## Local Development

```bash
# Install dependencies
pip install -r requirements.txt

# Download YOLO model (first time only)
# The model will be downloaded automatically on first run

# Run the API
python video_processor_api.py
```

The API will run on `http://localhost:9001`

## Railway Deployment

1. Connect this repository to Railway
2. Railway will automatically detect Python and install dependencies
3. The API will be available at `https://your-app.railway.app`

## Environment Variables

- `PORT`: Server port (default: 9001, Railway sets this automatically)
- `CORS_ORIGINS`: Allowed CORS origins (optional)

## API Endpoints

- `GET /`: Health check
- `GET /api/health`: Detailed health status
- `POST /api/process-video`: Process video file
- `GET /api/video/<video_id>`: Get processed video

## Notes

- Excel files (`Detected_Signals_Lat_Long.xlsx` or `Detected_Signals_Lat_Long_Enhanced.xlsx`) must be in the root directory
- YOLO model (`yolov8s.pt`) will be downloaded automatically on first run
- Video processing may take time depending on video length
