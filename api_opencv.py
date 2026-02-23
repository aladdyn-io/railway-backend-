"""
OpenCV API - Video Processing
Port: 5003
"""
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import os
import traceback
from werkzeug.utils import secure_filename

app = Flask(__name__)
CORS(app)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output'
os.makedirs(UPLOAD_FOLDER, exist_ok=True, mode=0o755)
os.makedirs(OUTPUT_FOLDER, exist_ok=True, mode=0o755)

ALLOWED_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/', methods=['GET'])
def root():
    return jsonify({
        'service': 'OpenCV API',
        'port': 5003,
        'endpoints': {
            '/video/info': 'POST - Get video information',
            '/video/extract_frame': 'POST - Extract frame at timestamp',
            '/video/extract_frames': 'POST - Extract multiple frames',
            '/health': 'GET - Health check'
        }
    })

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'healthy',
        'service': 'OpenCV API',
        'port': 5003
    })

@app.route('/video/info', methods=['POST'])
def video_info():
    """Get video information (duration, fps, frame count, etc.)"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type'}), 400
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400
        
        # Get video properties
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        duration = frame_count / fps if fps > 0 else 0
        
        cap.release()
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'video_info': {
                'filename': filename,
                'duration': duration,
                'fps': fps,
                'frame_count': frame_count,
                'width': width,
                'height': height
            }
        })
    except Exception as e:
        print(f"Error in video_info: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/video/extract_frame', methods=['POST'])
def extract_frame():
    """Extract a single frame at a specific timestamp"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        timestamp = float(request.form.get('timestamp', 0))
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_number = int(timestamp * fps)
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)
        
        ret, frame = cap.read()
        cap.release()
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        if not ret:
            return jsonify({'error': 'Could not extract frame'}), 400
        
        # Encode frame as JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame_base64 = np.frombuffer(buffer, np.uint8).tobytes()
        
        return jsonify({
            'success': True,
            'frame': frame_base64.hex(),  # Return as hex string
            'timestamp': timestamp,
            'frame_number': frame_number
        })
    except Exception as e:
        print(f"Error in extract_frame: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

@app.route('/video/extract_frames', methods=['POST'])
def extract_frames():
    """Extract multiple frames from video"""
    try:
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        file = request.files['video']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        num_frames = int(request.form.get('num_frames', 15))
        start_time = float(request.form.get('start_time', 0))
        end_time = float(request.form.get('end_time', -1))
        
        # Save uploaded file temporarily
        filename = secure_filename(file.filename)
        filepath = os.path.join(UPLOAD_FOLDER, filename)
        file.save(filepath)
        
        # Open video
        cap = cv2.VideoCapture(filepath)
        if not cap.isOpened():
            return jsonify({'error': 'Could not open video file'}), 400
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps if fps > 0 else 0
        
        if end_time < 0:
            end_time = duration
        
        # Calculate frame indices
        start_frame = int(start_time * fps)
        end_frame = int(end_time * fps)
        frame_indices = np.linspace(start_frame, end_frame, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                timestamp = frame_idx / fps
                _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
                frame_data = buffer.tobytes()
                frames.append({
                    'frame_number': int(frame_idx),
                    'timestamp': float(timestamp),
                    'data': frame_data.hex()  # Return as hex string
                })
        
        cap.release()
        
        # Clean up uploaded file
        try:
            os.remove(filepath)
        except:
            pass
        
        return jsonify({
            'success': True,
            'num_frames': len(frames),
            'frames': frames
        })
    except Exception as e:
        print(f"Error in extract_frames: {str(e)}")
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    print("Starting OpenCV API on port 5003...")
    app.run(host='0.0.0.0', port=5003, debug=True)
