from flask import Flask, request, render_template, send_file, redirect, url_for, session, jsonify
import os
import cv2
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timedelta
from dataclasses import dataclass
from typing import List, Dict, Any, Optional
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
import sqlite3
import threading
from pathlib import Path

# Import our enhanced analyzer
from enhanced_analyzer import EnhancedAnalyzer, AnalysisConfig, DetectionEvent

# Flask app setup
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configuration
UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'enhanced_analysis'
STATIC_FOLDER = 'static'
TEMPLATES_FOLDER = 'templates'

# Create directories
for folder in [UPLOAD_FOLDER, OUTPUT_FOLDER, STATIC_FOLDER]:
    os.makedirs(folder, exist_ok=True)

# User management
USERS = {
    'admin': generate_password_hash('railway2024', method='pbkdf2:sha256'),
    'operator': generate_password_hash('safety123', method='pbkdf2:sha256'),
    'supervisor': generate_password_hash('monitor456', method='pbkdf2:sha256')
}

# Global analyzer instance
analyzer = None
analysis_progress = {}

class AnalysisManager:
    """Manages analysis sessions and progress tracking"""
    
    def __init__(self):
        self.active_sessions = {}
        self.completed_analyses = {}
        
    def start_analysis(self, session_id: str, video_path: str, config: AnalysisConfig):
        """Start a new analysis session"""
        self.active_sessions[session_id] = {
            'status': 'starting',
            'progress': 0,
            'video_path': video_path,
            'config': config,
            'start_time': datetime.now(),
            'results': None
        }
        
        # Start analysis in background thread
        thread = threading.Thread(
            target=self._run_analysis,
            args=(session_id, video_path, config)
        )
        thread.daemon = True
        thread.start()
        
    def _run_analysis(self, session_id: str, video_path: str, config: AnalysisConfig):
        """Run analysis in background thread"""
        try:
            self.active_sessions[session_id]['status'] = 'processing'
            
            # Initialize analyzer
            analyzer = EnhancedAnalyzer(config)
            
            # Process video with progress updates
            results = analyzer.process_video_enhanced(video_path)
            
            # Store results
            self.active_sessions[session_id].update({
                'status': 'completed',
                'progress': 100,
                'results': results,
                'end_time': datetime.now()
            })
            
            # Move to completed analyses
            self.completed_analyses[session_id] = self.active_sessions[session_id]
            
        except Exception as e:
            self.active_sessions[session_id].update({
                'status': 'error',
                'error': str(e),
                'end_time': datetime.now()
            })
            
    def get_session_status(self, session_id: str) -> Dict[str, Any]:
        """Get status of analysis session"""
        if session_id in self.active_sessions:
            return self.active_sessions[session_id]
        elif session_id in self.completed_analyses:
            return self.completed_analyses[session_id]
        else:
            return {'status': 'not_found'}
            
    def get_all_completed(self) -> Dict[str, Any]:
        """Get all completed analyses"""
        return self.completed_analyses

# Initialize analysis manager
analysis_manager = AnalysisManager()

@app.route('/')
def index():
    """Main dashboard page"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get recent analyses
    recent_analyses = list(analysis_manager.get_all_completed().values())[-5:]
    
    return render_template('enhanced_index.html', 
                         recent_analyses=recent_analyses,
                         username=session['username'])

@app.route('/login', methods=['GET', 'POST'])
def login():
    """User authentication"""
    error = None
    if request.method == 'POST':
        username = request.form['username']
        password = request.form['password']
        
        if username in USERS and check_password_hash(USERS[username], password):
            session['username'] = username
            session['login_time'] = datetime.now().isoformat()
            return redirect(url_for('index'))
        else:
            error = 'Invalid credentials'
    
    return render_template('login.html', error=error)

@app.route('/logout')
def logout():
    """User logout"""
    session.clear()
    return redirect(url_for('login'))

@app.route('/upload', methods=['POST'])
def upload_video():
    """Handle video upload and start analysis"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    if 'video' not in request.files:
        return jsonify({'error': 'No video file provided'}), 400
    
    video_file = request.files['video']
    if video_file.filename == '':
        return jsonify({'error': 'No video file selected'}), 400
    
    # Get analysis configuration from form
    config = AnalysisConfig(
        pose_confidence=float(request.form.get('pose_confidence', 0.7)),
        hand_confidence=float(request.form.get('hand_confidence', 0.7)),
        face_confidence=float(request.form.get('face_confidence', 0.8)),
        object_confidence=float(request.form.get('object_confidence', 0.6)),
        enable_visualization=request.form.get('enable_visualization', 'true').lower() == 'true',
        generate_reports=request.form.get('generate_reports', 'true').lower() == 'true',
        output_folder=OUTPUT_FOLDER
    )
    
    # Save uploaded video
    filename = secure_filename(video_file.filename)
    timestamp = int(time.time())
    video_filename = f"{timestamp}_{filename}"
    video_path = os.path.join(UPLOAD_FOLDER, video_filename)
    video_file.save(video_path)
    
    # Generate session ID
    session_id = f"{session['username']}_{timestamp}"
    
    # Start analysis
    analysis_manager.start_analysis(session_id, video_path, config)
    
    return jsonify({
        'session_id': session_id,
        'status': 'started',
        'message': 'Analysis started successfully'
    })

@app.route('/analysis_status/<session_id>')
def analysis_status(session_id):
    """Get analysis progress and status"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    status = analysis_manager.get_session_status(session_id)
    
    # Remove sensitive data for client
    client_status = {
        'status': status.get('status', 'unknown'),
        'progress': status.get('progress', 0),
        'start_time': status.get('start_time', '').isoformat() if status.get('start_time') else '',
        'end_time': status.get('end_time', '').isoformat() if status.get('end_time') else '',
        'error': status.get('error', '')
    }
    
    return jsonify(client_status)

@app.route('/results/<session_id>')
def view_results(session_id):
    """View analysis results"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    session_data = analysis_manager.get_session_status(session_id)
    
    if session_data['status'] != 'completed':
        return render_template('analysis_progress.html', 
                             session_id=session_id,
                             status=session_data['status'])
    
    results = session_data['results']
    
    # Prepare data for dashboard
    stats = results['statistics']
    timeline = results['timeline']
    
    # Group events by type for the dashboard
    events_by_type = results.get('events_by_type', {})
    
    # Convert events to dashboard format
    sleep_events = []
    mobile_events = []
    gesture_events = []
    
    for event_type, events in events_by_type.items():
        for event in events:
            event_data = {
                'time': f"{int(event.timestamp//60):02d}:{int(event.timestamp%60):02d}",
                'confidence': event.confidence,
                'quality': event.quality_score,
                'image': os.path.basename(event.frame_path) if event.frame_path else None
            }
            
            if 'drowsiness' in event_type.lower():
                event_data['reason'] = event.metadata.get('detection_method', 'Multi-indicator')
                sleep_events.append(event_data)
            elif 'mobile' in event_type.lower():
                event_data['action'] = 'Phone Usage Detected'
                mobile_events.append(event_data)
            elif 'hand' in event_type.lower() or 'gesture' in event_type.lower():
                event_data['action'] = f"Signal: {event.metadata.get('gesture_type', 'Hand Raised')}"
                gesture_events.append(event_data)
    
    # Enhanced statistics
    enhanced_stats = {
        'total_sleep': len(sleep_events),
        'total_mobile': len(mobile_events),
        'total_gestures': len(gesture_events),
        'duration': stats['video_duration'],
        'processing_time': stats['processing_time'],
        'accuracy': stats.get('average_confidence', 0) * 100,
        'total_events': stats['total_events']
    }
    
    return render_template('enhanced_dashboard.html',
                         stats=enhanced_stats,
                         timeline=timeline,
                         sleep_events=sleep_events,
                         mobile_events=mobile_events,
                         gesture_events=gesture_events,
                         session_id=session_id)

@app.route('/api/analytics/<session_id>')
def get_analytics(session_id):
    """API endpoint for analytics data"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    session_data = analysis_manager.get_session_status(session_id)
    
    if session_data['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    results = session_data['results']
    
    # Prepare analytics data
    analytics = {
        'performance': results['performance'],
        'quality_metrics': results['quality_metrics'],
        'timeline_data': results['timeline'],
        'statistics': results['statistics']
    }
    
    return jsonify(analytics)

@app.route('/export_report/<session_id>')
def export_report(session_id):
    """Export analysis report"""
    if 'username' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    
    session_data = analysis_manager.get_session_status(session_id)
    
    if session_data['status'] != 'completed':
        return jsonify({'error': 'Analysis not completed'}), 400
    
    results = session_data['results']
    
    # Create comprehensive report
    report_data = {
        'analysis_info': {
            'session_id': session_id,
            'analyst': session['username'],
            'video_path': session_data['video_path'],
            'analysis_date': session_data['start_time'].isoformat(),
            'processing_time': session_data['end_time'] - session_data['start_time']
        },
        'statistics': results['statistics'],
        'timeline': results['timeline'],
        'performance': results['performance'],
        'quality_metrics': results['quality_metrics']
    }
    
    # Save report as JSON
    report_filename = f"analysis_report_{session_id}.json"
    report_path = os.path.join(OUTPUT_FOLDER, report_filename)
    
    with open(report_path, 'w') as f:
        json.dump(report_data, f, indent=2, default=str)
    
    return send_file(report_path, as_attachment=True, download_name=report_filename)

@app.route('/images/<filename>')
def serve_image(filename):
    """Serve detection images"""
    image_path = os.path.join(OUTPUT_FOLDER, 'frames', filename)
    if os.path.exists(image_path):
        return send_file(image_path)
    else:
        # Return placeholder image
        return send_file(os.path.join(STATIC_FOLDER, 'placeholder.jpg'))

@app.route('/dashboard')
def dashboard():
    """Main analytics dashboard"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    # Get all completed analyses for overview
    all_analyses = analysis_manager.get_all_completed()
    
    # Calculate aggregate statistics
    total_analyses = len(all_analyses)
    total_events = sum(a['results']['statistics']['total_events'] 
                      for a in all_analyses.values() 
                      if a['status'] == 'completed')
    
    avg_processing_time = np.mean([a['results']['statistics']['processing_time'] 
                                  for a in all_analyses.values() 
                                  if a['status'] == 'completed']) if all_analyses else 0
    
    dashboard_stats = {
        'total_analyses': total_analyses,
        'total_events': total_events,
        'avg_processing_time': avg_processing_time,
        'active_sessions': len(analysis_manager.active_sessions)
    }
    
    return render_template('dashboard_overview.html',
                         stats=dashboard_stats,
                         recent_analyses=list(all_analyses.values())[-10:])

@app.route('/settings', methods=['GET', 'POST'])
def settings():
    """Analysis settings configuration"""
    if 'username' not in session:
        return redirect(url_for('login'))
    
    if request.method == 'POST':
        # Save user preferences
        preferences = {
            'pose_confidence': float(request.form.get('pose_confidence', 0.7)),
            'hand_confidence': float(request.form.get('hand_confidence', 0.7)),
            'face_confidence': float(request.form.get('face_confidence', 0.8)),
            'object_confidence': float(request.form.get('object_confidence', 0.6)),
            'enable_visualization': request.form.get('enable_visualization') == 'on',
            'generate_reports': request.form.get('generate_reports') == 'on'
        }
        
        # Store in session
        session['preferences'] = preferences
        
        return jsonify({'status': 'success', 'message': 'Settings saved'})
    
    # Get current preferences
    preferences = session.get('preferences', {
        'pose_confidence': 0.7,
        'hand_confidence': 0.7,
        'face_confidence': 0.8,
        'object_confidence': 0.6,
        'enable_visualization': True,
        'generate_reports': True
    })
    
    return render_template('settings.html', preferences=preferences)

@app.errorhandler(404)
def not_found(error):
    return render_template('error.html', 
                         error_code=404, 
                         error_message="Page not found"), 404

@app.errorhandler(500)
def internal_error(error):
    return render_template('error.html', 
                         error_code=500, 
                         error_message="Internal server error"), 500

# Template filters
@app.template_filter('tojsonfilter')
def to_json_filter(obj):
    return json.dumps(obj)

if __name__ == '__main__':
    # Create necessary directories
    os.makedirs('templates', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    # Run the application
    app.run(debug=True, host='0.0.0.0', port=8080, threaded=True)