"""
MediaPipe API - Hand, Face, and Pose Detection
Port: 5001
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from mediapipe import solutions as mp_solutions
import traceback

app = Flask(__name__)
CORS(app)

# Initialize MediaPipe solutions (create new instances per request to avoid timestamp issues)
mp_hands = mp_solutions.hands
mp_pose = mp_solutions.pose
mp_face_mesh = mp_solutions.face_mesh
mp_face_detection = mp_solutions.face_detection

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'MediaPipe API',
        'port': 5001,
        'endpoints': {
            '/detect/hands': 'POST - Detect hand landmarks',
            '/detect/pose': 'POST - Detect pose landmarks',
            '/detect/hand_signal': 'POST - Detect raised hand signals',
            '/detect/face': 'POST - Detect face landmarks',
            '/detect/drowsiness': 'POST - Detect drowsiness/microsleep',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'healthy', 'service': 'MediaPipe API', 'port': 5001})

@app.route('/detect/hands', methods=['POST'])
def detect_hands():
    """Detect hand landmarks in an image"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read and decode image
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not decode image'}), 400
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create new instance for each request
        with mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            results = hands.process(rgb_image)
            
            detections = []
            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    landmarks = [
                        {'x': float(lm.x), 'y': float(lm.y), 'z': float(lm.z)}
                        for lm in hand_landmarks.landmark
                    ]
                    hand_type = results.multi_handedness[idx].classification[0].label
                    detections.append({
                        'hand_id': idx,
                        'hand_type': hand_type,
                        'landmarks': landmarks
                    })
            
            return jsonify({
                'success': True,
                'num_hands': len(detections),
                'detections': detections
            })
    except Exception as e:
        print(f"Error in detect_hands: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/detect/pose', methods=['POST'])
def detect_pose():
    """Detect pose landmarks in an image"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not decode image'}), 400
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create new instance for each request
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            smooth_landmarks=True
        ) as pose:
            results = pose.process(rgb_image)
            
            landmarks = []
            if results.pose_landmarks:
                landmarks = [
                    {
                        'x': float(lm.x),
                        'y': float(lm.y),
                        'z': float(lm.z),
                        'visibility': float(lm.visibility)
                    }
                    for lm in results.pose_landmarks.landmark
                ]
            
            return jsonify({
                'success': True,
                'detected': len(landmarks) > 0,
                'landmarks': landmarks
            })
    except Exception as e:
        print(f"Error in detect_pose: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/detect/hand_signal', methods=['POST'])
def detect_hand_signal():
    """Detect raised hand signals"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not decode image'}), 400
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create new instances for each request
        with mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            min_detection_confidence=0.5,
            smooth_landmarks=True
        ) as pose, \
        mp_hands.Hands(
            static_image_mode=True,
            max_num_hands=2,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as hands:
            pose_results = pose.process(rgb_image)
            hand_results = hands.process(rgb_image)
            
            signals_detected = []
            if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                left_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
                right_shoulder = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
                nose = pose_results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
                
                for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                    wrist = hand_landmarks.landmark[mp_hands.HandLandmark.WRIST]
                    hand_raised = wrist.y < min(left_shoulder.y, right_shoulder.y)
                    hand_above_head = wrist.y < nose.y
                    
                    if hand_raised or hand_above_head:
                        signals_detected.append({
                            'hand_id': idx,
                            'hand_raised': hand_raised,
                            'hand_above_head': hand_above_head
                        })
            
            return jsonify({
                'success': True,
                'signal_detected': len(signals_detected) > 0,
                'detections': signals_detected
            })
    except Exception as e:
        print(f"Error in detect_hand_signal: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/detect/face', methods=['POST'])
def detect_face():
    """Detect face landmarks"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not decode image'}), 400
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Create new instance for each request
        with mp_face_detection.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5
        ) as face_detection:
            results = face_detection.process(rgb_image)
            
            detections = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box
                    detections.append({
                        'confidence': float(detection.score[0]),
                        'bbox': {
                            'x': float(bbox.xmin),
                            'y': float(bbox.ymin),
                            'width': float(bbox.width),
                            'height': float(bbox.height)
                        }
                    })
            
            return jsonify({
                'success': True,
                'num_faces': len(detections),
                'detections': detections
            })
    except Exception as e:
        print(f"Error in detect_face: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/detect/drowsiness', methods=['POST'])
def detect_drowsiness():
    """Detect drowsiness/microsleep using eye aspect ratio and head pose"""
    try:
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        file_data = file.read()
        if len(file_data) == 0:
            return jsonify({'success': False, 'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'success': False, 'error': 'Could not decode image'}), 400
        
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w = image.shape[:2]
        
        # Eye aspect ratio threshold
        EYE_AR_THRESH = 0.27
        
        # Left and right eye landmark indices
        LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        
        def eye_aspect_ratio(landmarks, indices):
            points = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
            A = np.linalg.norm(points[1] - points[5])
            B = np.linalg.norm(points[2] - points[4])
            C = np.linalg.norm(points[0] - points[3])
            if C == 0:
                return 1.0
            return (A + B) / (2.0 * C)
        
        # Create new instance for each request
        with mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        ) as face_mesh:
            results = face_mesh.process(rgb_image)
            
            drowsy = False
            reason = None
            ear = None
            
            if results.multi_face_landmarks:
                landmarks = results.multi_face_landmarks[0].landmark
                left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX)
                right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX)
                avg_ear = (left_ear + right_ear) / 2.0
                ear = avg_ear
                
                if avg_ear < EYE_AR_THRESH:
                    drowsy = True
                    reason = 'low_eye_aspect_ratio'
            
            return jsonify({
                'success': True,
                'drowsy': drowsy,
                'ear': ear,
                'reason': reason
            })
    except Exception as e:
        print(f"Error in detect_drowsiness: {str(e)}")
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting MediaPipe API on port 5001...")
    app.run(host='0.0.0.0', port=5001, debug=True)
