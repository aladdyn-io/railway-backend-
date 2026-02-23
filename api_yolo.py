"""
YOLO API - Object, Phone, and Bag Detection
Port: 5002
"""
from flask import Flask, request, jsonify
from flask_cors import CORS
import cv2
import numpy as np
from ultralytics import YOLO
import traceback
import os

app = Flask(__name__)
CORS(app)

# Load YOLO model
MODEL_PATH = 'yolov8s.pt'
if not os.path.exists(MODEL_PATH):
    print(f"Warning: {MODEL_PATH} not found. YOLO will download it on first use.")

try:
    yolo_model = YOLO(MODEL_PATH)
    print(f"YOLO model loaded: {MODEL_PATH}")
except Exception as e:
    print(f"Error loading YOLO model: {e}")
    yolo_model = None

# Class IDs for detection
PHONE_CLASS = 67  # cell phone in COCO dataset
BAG_CLASSES = [24, 26, 28]  # backpack, handbag, suitcase in COCO dataset

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'YOLO API',
        'port': 5002,
        'model': MODEL_PATH,
        'endpoints': {
            '/detect/objects': 'POST - Detect all objects',
            '/detect/phone': 'POST - Detect phones',
            '/detect/bags': 'POST - Detect bags/backpacks',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    model_status = 'loaded' if yolo_model is not None else 'not_loaded'
    return jsonify({
        'status': 'healthy',
        'service': 'YOLO API',
        'port': 5002,
        'model': model_status
    })

@app.route('/detect/objects', methods=['POST'])
def detect_objects():
    """Detect all objects in an image"""
    try:
        if yolo_model is None:
            return jsonify({'error': 'YOLO model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        confidence = float(request.form.get('confidence', 0.25))
        file_data = file.read()
        
        if len(file_data) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Run prediction
        results = yolo_model.predict(image, verbose=False, conf=confidence)[0]
        detections = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                try:
                    # Extract values from tensors
                    cls_id = int(box.cls[0].item() if hasattr(box.cls[0], 'item') else box.cls[0])
                    conf = float(box.conf[0].item() if hasattr(box.conf[0], 'item') else box.conf[0])
                    xyxy = box.xyxy[0].cpu().numpy()
                    x1, y1, x2, y2 = map(int, xyxy)
                    
                    detections.append({
                        'class_id': cls_id,
                        'class_name': yolo_model.names[cls_id],
                        'confidence': conf,
                        'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                    })
                except Exception as box_error:
                    print(f"Error processing box: {box_error}")
                    continue
        
        return jsonify({
            'success': True,
            'num_detections': len(detections),
            'detections': detections
        })
    except Exception as e:
        print(f"Error in detect_objects: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/detect/phone', methods=['POST'])
def detect_phone():
    """Detect phones in an image"""
    try:
        if yolo_model is None:
            return jsonify({'error': 'YOLO model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        confidence = float(request.form.get('confidence', 0.25))
        file_data = file.read()
        
        if len(file_data) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Run prediction
        results = yolo_model.predict(image, verbose=False, conf=confidence)[0]
        phones = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                try:
                    # Extract values from tensors - same pattern as detect_objects
                    cls_id = int(box.cls[0].item() if hasattr(box.cls[0], 'item') else box.cls[0])
                    conf = float(box.conf[0].item() if hasattr(box.conf[0], 'item') else box.conf[0])
                    
                    if cls_id == PHONE_CLASS:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        phones.append({
                            'confidence': conf,
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                        })
                except Exception as box_error:
                    import traceback
                    print(f"Error processing box in detect_phone: {box_error}")
                    traceback.print_exc()
                    continue
        
        return jsonify({
            'success': True,
            'phone_detected': len(phones) > 0,
            'num_phones': len(phones),
            'detections': phones
        })
    except Exception as e:
        error_msg = str(e)
        print(f"Error in detect_phone: {error_msg}")
        traceback.print_exc()
        # Return full error message, not truncated
        return jsonify({
            'success': False,
            'error': error_msg,
            'phone_detected': False,
            'num_phones': 0,
            'detections': []
        }), 500

@app.route('/detect/bags', methods=['POST'])
def detect_bags():
    """Detect bags/backpacks in an image"""
    try:
        if yolo_model is None:
            return jsonify({'error': 'YOLO model not loaded'}), 500
        
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        confidence = float(request.form.get('confidence', 0.25))
        file_data = file.read()
        
        if len(file_data) == 0:
            return jsonify({'error': 'Empty file'}), 400
        
        image = cv2.imdecode(np.frombuffer(file_data, np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            return jsonify({'error': 'Could not decode image'}), 400
        
        # Run prediction
        results = yolo_model.predict(image, verbose=False, conf=confidence)[0]
        bags = []
        
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                try:
                    # Extract values from tensors - same pattern as detect_objects
                    cls_id = int(box.cls[0].item() if hasattr(box.cls[0], 'item') else box.cls[0])
                    conf = float(box.conf[0].item() if hasattr(box.conf[0], 'item') else box.conf[0])
                    class_name = yolo_model.names[cls_id]
                    
                    if cls_id in BAG_CLASSES:
                        xyxy = box.xyxy[0].cpu().numpy()
                        x1, y1, x2, y2 = map(int, xyxy)
                        bags.append({
                            'class_name': class_name,
                            'confidence': conf,
                            'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2}
                        })
                except Exception as box_error:
                    import traceback
                    print(f"Error processing box in detect_bags: {box_error}")
                    traceback.print_exc()
                    continue
        
        return jsonify({
            'success': True,
            'bags_detected': len(bags) > 0,
            'num_bags': len(bags),
            'detections': bags
        })
    except Exception as e:
        error_msg = str(e)
        print(f"Error in detect_bags: {error_msg}")
        traceback.print_exc()
        # Return full error message, not truncated
        return jsonify({
            'success': False,
            'error': error_msg,
            'bags_detected': False,
            'num_bags': 0,
            'detections': []
        }), 500

if __name__ == '__main__':
    print("Starting YOLO API on port 5002...")
    app.run(host='0.0.0.0', port=5002, debug=True)
