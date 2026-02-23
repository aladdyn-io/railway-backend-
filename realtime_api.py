from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import os
from ultralytics import YOLO
from mediapipe import solutions as mp_solutions

app = Flask(__name__)
CORS(app)

# Initialize models
yolo = YOLO('yolov8s.pt')
face_mesh = mp_solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
hands = mp_solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
holistic = mp_solutions.holistic.Holistic(min_detection_confidence=0.5)

def decode_image(image_data):
    """Decode base64 image to OpenCV format"""
    image_bytes = base64.b64decode(image_data.split(',')[1])
    nparr = np.frombuffer(image_bytes, np.uint8)
    return cv2.imdecode(nparr, cv2.IMREAD_COLOR)

@app.route('/api/hand-signal', methods=['POST'])
def detect_hand_signal():
    try:
        data = request.json
        frame = decode_image(data['image'])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with holistic
        results = holistic.process(rgb_frame)
        
        detected = False
        signal_type = "none"
        
        if results.pose_landmarks and results.left_hand_landmarks:
            # Check elbow above shoulder
            left_shoulder = results.pose_landmarks.landmark[11].y
            left_elbow = results.pose_landmarks.landmark[13].y
            wrist_y = results.left_hand_landmarks.landmark[0].y
            
            if left_elbow < left_shoulder and wrist_y < 0.4:
                detected = True
                signal_type = "hand_raised"
        
        return jsonify({
            "detected": detected,
            "signal_type": signal_type,
            "confidence": 0.85 if detected else 0.0,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/mobile-phone', methods=['POST'])
def detect_mobile_phone():
    try:
        data = request.json
        frame = decode_image(data['image'])
        
        # YOLO detection
        results = yolo.predict(frame, verbose=False, conf=0.4)[0]
        
        detected = False
        confidence = 0.0
        
        for box in results.boxes:
            if int(box.cls[0]) == 67:  # Phone class
                detected = True
                confidence = float(box.conf[0])
                break
        
        return jsonify({
            "detected": detected,
            "confidence": confidence,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/microsleep', methods=['POST'])
def detect_microsleep():
    try:
        data = request.json
        frame = decode_image(data['image'])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Face mesh detection
        results = face_mesh.process(rgb_frame)
        
        detected = False
        reason = "none"
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            
            # Eye aspect ratio calculation
            left_eye_idx = [33, 160, 158, 133, 153, 144]
            right_eye_idx = [362, 385, 387, 263, 373, 380]
            
            def eye_aspect_ratio(indices):
                pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
                A = np.linalg.norm(pts[1] - pts[5])
                B = np.linalg.norm(pts[2] - pts[4])
                C = np.linalg.norm(pts[0] - pts[3])
                return (A + B) / (2.0 * C)
            
            left_ear = eye_aspect_ratio(left_eye_idx)
            right_ear = eye_aspect_ratio(right_eye_idx)
            avg_ear = (left_ear + right_ear) / 2.0
            
            if avg_ear < 0.27:
                detected = True
                reason = "drowsy_eyes"
        
        return jsonify({
            "detected": detected,
            "reason": reason,
            "confidence": 0.9 if detected else 0.0,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/packing', methods=['POST'])
def detect_packing():
    try:
        data = request.json
        frame = decode_image(data['image'])
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Detect bags with YOLO
        yolo_results = yolo.predict(frame, verbose=False, conf=0.4)[0]
        bag_boxes = []
        for box in yolo_results.boxes:
            if int(box.cls[0]) in [24, 25, 26]:  # Bag classes
                bag_boxes.append(box.xyxy[0].cpu().numpy())
        
        # Detect hands
        hand_results = hands.process(rgb_frame)
        
        detected = False
        action = "none"
        
        if hand_results.multi_hand_landmarks and bag_boxes:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                h, w = frame.shape[:2]
                cx = int(np.mean([lm.x * w for lm in hand_landmarks.landmark]))
                cy = int(np.mean([lm.y * h for lm in hand_landmarks.landmark]))
                
                # Check if hand is near any bag
                for x1, y1, x2, y2 in bag_boxes:
                    bag_cx, bag_cy = (x1 + x2) // 2, (y1 + y2) // 2
                    if (cx - bag_cx)**2 + (cy - bag_cy)**2 <= 6400:  # 80px threshold
                        detected = True
                        action = "packing_detected"
                        break
        
        return jsonify({
            "detected": detected,
            "action": action,
            "confidence": 0.8 if detected else 0.0,
            "timestamp": datetime.now().isoformat()
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy", "timestamp": datetime.now().isoformat()})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=9000, debug=True)