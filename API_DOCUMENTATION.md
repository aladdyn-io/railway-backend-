# Real-time Detection APIs

## Server Setup
```bash
cd "/Users/yakesh/Downloads/Railways Project"
python3 realtime_api.py
```

Server will run on: http://localhost:9000

## API Endpoints

### 1. Hand Signal Detection
**POST** `/api/hand-signal`
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "detected": true,
  "signal_type": "hand_raised",
  "confidence": 0.85,
  "timestamp": "2024-01-12T10:30:00"
}
```

### 2. Mobile Phone Detection
**POST** `/api/mobile-phone`
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "detected": true,
  "confidence": 0.92,
  "timestamp": "2024-01-12T10:30:00"
}
```

### 3. Microsleep Detection
**POST** `/api/microsleep`
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "detected": true,
  "reason": "drowsy_eyes",
  "confidence": 0.9,
  "timestamp": "2024-01-12T10:30:00"
}
```

### 4. Packing/Unpacking Detection
**POST** `/api/packing`
```json
{
  "image": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQAAAQ..."
}
```

**Response:**
```json
{
  "detected": true,
  "action": "packing_detected",
  "confidence": 0.8,
  "timestamp": "2024-01-12T10:30:00"
}
```

### 5. Health Check
**GET** `/api/health`

**Response:**
```json
{
  "status": "healthy",
  "timestamp": "2024-01-12T10:30:00"
}
```

## Frontend Integration

Your frontend at `http://localhost:8083/dashboard` can call these APIs by:

1. Capturing video frames from webcam
2. Converting frames to base64
3. Sending POST requests to the respective endpoints
4. Processing the JSON responses

## Testing

Run the test script:
```bash
python3 test_api.py
```

This will test all endpoints with a webcam capture.