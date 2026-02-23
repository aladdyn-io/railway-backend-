# 🚀 API Integration Instructions

## ✅ All 4 APIs Ready for Integration

Your 4 separate APIs are ready:
- **MediaPipe API** (Port 5001) - `api_mediapipe.py`
- **YOLO API** (Port 5002) - `api_yolo.py`
- **OpenCV API** (Port 5003) - `api_opencv.py`
- **Flask Web App** (Port 5000) - `app.py`

---

## 📋 Steps to Integrate

### Step 1: Clone the Repository

If the repository is private, you'll need to authenticate first:

```bash
cd "/Users/yakesh/Downloads/Railways Project"
git clone https://github.com/vijayasvj/pilot-eye-analytics-hub.git
```

Or if you have SSH access:
```bash
git clone git@github.com:vijayasvj/pilot-eye-analytics-hub.git
```

### Step 2: Copy Integration Files

Copy the files from `integration_files/` to your project:

1. **API Service** - Copy `api-service.ts` to your project's `src/services/` or similar
2. **Update components** - Use the API service in your video components

### Step 3: Start All APIs

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

**Terminal 4 (Optional):**
```bash
python3 app.py
```

---

## 🔌 API Endpoints Available

### MediaPipe API (http://localhost:5001)
- `GET /` - API info
- `GET /health` - Health check
- `POST /detect/hands` - Detect hands
- `POST /detect/pose` - Detect pose
- `POST /detect/hand_signal` - Detect hand signals

### YOLO API (http://localhost:5002)
- `GET /` - API info
- `GET /health` - Health check
- `POST /detect/objects` - Detect all objects
- `POST /detect/phone` - Detect phones
- `POST /detect/bags` - Detect bags

### OpenCV API (http://localhost:5003)
- `GET /` - API info
- `GET /health` - Health check
- `POST /video/info` - Get video metadata
- `POST /video/extract_frame` - Extract frame

---

## 💡 Usage Example

```typescript
import { processFrame, checkAPIsHealth } from '@/services/api';

// Check API health
const health = await checkAPIsHealth();
console.log(health); // { mediapipe: true, yolo: true, opencv: true }

// Process a video frame
const imageFile = // ... get from video element
const result = await processFrame(imageFile);

if (result.handSignal?.signal_detected) {
  console.log('Hand signal detected!');
}
if (result.phone?.phone_detected) {
  console.log('Phone detected!');
}
if (result.bags?.bags_detected) {
  console.log('Bag detected!');
}
```

---

## 📝 Next Steps

Once you clone the repository, I can:
1. Explore the project structure
2. Integrate the APIs into the correct components
3. Update the UI to use real-time detection
4. Test everything together

**Let me know once you've cloned the repository!**



