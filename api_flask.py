"""
Flask Web API - Main Interface
Port: 5000
This is the main web interface that coordinates all other APIs
"""
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import requests
import traceback

app = Flask(__name__)
CORS(app)

# API endpoints
MEDIAPIPE_API = 'http://localhost:5001'
YOLO_API = 'http://localhost:5002'
OPENCV_API = 'http://localhost:5003'

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'Flask Web API',
        'port': 5000,
        'description': 'Main web interface coordinating all detection APIs',
        'apis': {
            'mediapipe': MEDIAPIPE_API,
            'yolo': YOLO_API,
            'opencv': OPENCV_API
        },
        'endpoints': {
            '/health': 'GET - Health check',
            '/api/status': 'GET - Check all API statuses'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'Flask Web API',
        'port': 5000
    })

@app.route('/api/status', methods=['GET'])
def api_status():
    """Check status of all APIs"""
    status = {
        'mediapipe': {'status': 'unknown', 'url': MEDIAPIPE_API},
        'yolo': {'status': 'unknown', 'url': YOLO_API},
        'opencv': {'status': 'unknown', 'url': OPENCV_API}
    }
    
    # Check MediaPipe API
    try:
        response = requests.get(f'{MEDIAPIPE_API}/health', timeout=2)
        status['mediapipe']['status'] = 'online' if response.status_code == 200 else 'offline'
    except:
        status['mediapipe']['status'] = 'offline'
    
    # Check YOLO API
    try:
        response = requests.get(f'{YOLO_API}/health', timeout=2)
        status['yolo']['status'] = 'online' if response.status_code == 200 else 'offline'
    except:
        status['yolo']['status'] = 'offline'
    
    # Check OpenCV API
    try:
        response = requests.get(f'{OPENCV_API}/health', timeout=2)
        status['opencv']['status'] = 'online' if response.status_code == 200 else 'offline'
    except:
        status['opencv']['status'] = 'offline'
    
    return jsonify({
        'success': True,
        'apis': status
    })

if __name__ == '__main__':
    print("Starting Flask Web API on port 5000...")
    print(f"MediaPipe API: {MEDIAPIPE_API}")
    print(f"YOLO API: {YOLO_API}")
    print(f"OpenCV API: {OPENCV_API}")
    app.run(host='0.0.0.0', port=5000, debug=True)



