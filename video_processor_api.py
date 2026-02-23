from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import cv2
import numpy as np
import base64
import json
from datetime import datetime
import os
import uuid
from ultralytics import YOLO
from mediapipe import solutions as mp_solutions
from perfect_hand_signal_detector import PerfectHandSignalDetector
from station_alert_system import StationAlertSystem
from requirement_team_loader import (
    load_gdr_mas_rules,
    load_gps_log,
    load_signal_rules_from_bytes,
    correlate_sid_with_rtis,
    assign_time_windows,
    validate_station_alerts,
    get_rules_for_frontend
)
try:
    from video_timestamp_ocr import extract_time_from_frame
    _OCR_AVAILABLE = True
except ImportError:
    _OCR_AVAILABLE = False

app = Flask(__name__)
CORS(app)

# Initialize models
yolo = YOLO('yolov8s.pt')
face_mesh = mp_solutions.face_mesh.FaceMesh(max_num_faces=1, min_detection_confidence=0.5)
hands = mp_solutions.hands.Hands(max_num_hands=2, min_detection_confidence=0.5)
holistic = mp_solutions.holistic.Holistic(min_detection_confidence=0.5)

class VideoProcessor:
    def __init__(self):
        self.EYE_AR_THRESH = 0.27
        self.LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
        self.RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
        
        # Initialize Perfect Hand Signal Detector with Excel rules
        # Try enhanced file first, fallback to original
        excel_file = "Detected_Signals_Lat_Long_Enhanced.xlsx"
        import os
        if not os.path.exists(excel_file):
            excel_file = "Detected_Signals_Lat_Long.xlsx"
            print(f"⚠️ Using basic Excel file: {excel_file}")
            print("💡 Run 'python3 enhance_station_rules.py' to create enhanced file with timing")
        else:
            print(f"✅ Using enhanced Excel file: {excel_file}")
        
        self.perfect_hand_detector = PerfectHandSignalDetector(excel_file)
        
        # Initialize Station Alert System
        self.station_alert_system = StationAlertSystem(excel_file)
        # Bag packing: require sustained interaction (not walking past)
        self._bag_packing_streak = 0

    def eye_aspect_ratio(self, landmarks, indices, w, h):
        pts = np.array([(landmarks[i].x * w, landmarks[i].y * h) for i in indices])
        if len(pts) < 6: return 1.0
        A = np.linalg.norm(pts[1] - pts[5])
        B = np.linalg.norm(pts[2] - pts[4])
        C = np.linalg.norm(pts[0] - pts[3])
        return (A + B) / (2.0 * C)

    def _is_hand_near_bbox(self, wrist_norm_x, wrist_norm_y, bbox, w, h, margin_ratio=0.05):
        """Check if wrist (normalized 0-1) is inside bag bbox = pilot packing (not just walking past)."""
        wx = wrist_norm_x * w
        wy = wrist_norm_y * h
        x1, y1, x2, y2 = bbox
        bw, bh = x2 - x1, y2 - y1
        margin = max(bw, bh) * margin_ratio
        return (x1 - margin <= wx <= x2 + margin) and (y1 - margin <= wy <= y2 + margin)
    
    def process_frame(self, frame, timestamp_seconds):
        h, w = frame.shape[:2]
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        detections = {
            'timestamp_seconds': timestamp_seconds,
            'phone': {'phone_detected': False, 'detections': []},
            'handSignal': {'signal_detected': False, 'signal_type': 'none'},
            'microsleep': {'detected': False, 'reason': 'none'},
            'bags': {'bags_detected': False, 'detections': []},
            'objects': []
        }
        
        # YOLO Detection - Higher accuracy with confidence filtering
        bag_candidates = []  # Only report bags when pilot is packing (hand near bag)
        yolo_results = yolo.predict(frame, verbose=False, conf=0.10)[0]  # Lower initial threshold
        for box in yolo_results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
            
            class_name = yolo.names[cls_id]
            detections['objects'].append({
                'class_name': class_name,
                'confidence': conf,
                'bbox': [int(x1), int(y1), int(x2), int(y2)]
            })
            
            # Enhanced phone detection with higher confidence
            if cls_id == 67 and conf >= 0.25:  # Stricter phone detection
                detections['phone']['phone_detected'] = True
                detections['phone']['detections'].append({
                    'confidence': conf,
                    'bbox': [int(x1), int(y1), int(x2), int(y2)]
                })
                print(f"📱 PHONE DETECTED! Confidence: {conf:.3f}")
            
            # Bag candidates (only count when pilot is packing = hand near bag)
            elif cls_id in [24, 26, 28] and conf >= 0.20:  # backpack(24), handbag(26), suitcase(28)
                bbox_area = (x2 - x1) * (y2 - y1)
                if bbox_area > 2000:  # Filter small detections
                    bag_candidates.append({
                        'class_name': class_name,
                        'confidence': conf,
                        'bbox': [int(x1), int(y1), int(x2), int(y2)]
                    })
            # Also check for laptop as potential phone substitute
            elif cls_id == 63:  # laptop
                print(f"💻 LAPTOP DETECTED! (might be mistaken for phone) Confidence: {conf:.2f}")

        # Bag detection: only when pilot is PACKING (hand inside bag, sustained for 2+ frames)
        if bag_candidates:
            holistic_result = holistic.process(rgb_frame)
            wrist_positions = []  # [(x_norm, y_norm), ...] only visible wrists
            if holistic_result.pose_landmarks:
                lm = holistic_result.pose_landmarks.landmark
                left_wrist = lm[mp_solutions.pose.PoseLandmark.LEFT_WRIST]
                right_wrist = lm[mp_solutions.pose.PoseLandmark.RIGHT_WRIST]
                visibility_threshold = 0.5
                if getattr(left_wrist, 'visibility', 1.0) >= visibility_threshold:
                    wrist_positions.append((left_wrist.x, left_wrist.y))
                if getattr(right_wrist, 'visibility', 1.0) >= visibility_threshold:
                    wrist_positions.append((right_wrist.x, right_wrist.y))
            hand_near_any = False
            for bag in bag_candidates:
                hand_near = any(
                    self._is_hand_near_bbox(wx, wy, bag['bbox'], w, h)
                    for wx, wy in wrist_positions
                )
                if hand_near:
                    hand_near_any = True
                    break
            if hand_near_any:
                self._bag_packing_streak += 1
                if self._bag_packing_streak >= 2:  # Sustained interaction, not walking past
                    for bag in bag_candidates:
                        hand_near = any(
                            self._is_hand_near_bbox(wx, wy, bag['bbox'], w, h)
                            for wx, wy in wrist_positions
                        )
                        if hand_near:
                            detections['bags']['bags_detected'] = True
                            detections['bags']['detections'].append({
                                'confidence': bag['confidence'],
                                'bbox': bag['bbox']
                            })
                    if detections['bags']['bags_detected']:
                        print(f"🎒 PACKING DETECTED! Pilot actively packing — Confidence: {detections['bags']['detections'][0]['confidence']:.3f}")
            else:
                self._bag_packing_streak = 0
        else:
            self._bag_packing_streak = 0

        # Perfect Hand Signal Detection with Excel Rules Compliance
        try:
            hand_signal_detection = self.perfect_hand_detector.process_frame(frame, timestamp_seconds)
            
            if hand_signal_detection:
                detections['handSignal']['signal_detected'] = True
                detections['handSignal']['signal_type'] = hand_signal_detection.signal_type
                detections['handSignal']['compliance_validation'] = {
                    'compliance_status': hand_signal_detection.compliance_status,
                    'violation_type': hand_signal_detection.violation_type,
                    'confidence': hand_signal_detection.confidence,
                    'current_gps': hand_signal_detection.current_gps,
                    'nearest_station': hand_signal_detection.nearest_rule.station_name if hand_signal_detection.nearest_rule else None,
                    'distance_to_station': hand_signal_detection.distance_to_rule,
                    'tolerance_meters': hand_signal_detection.nearest_rule.tolerance_meters if hand_signal_detection.nearest_rule else None
                }
                
                if hand_signal_detection.compliance_status == "COMPLIANT":
                    print(f"✅ PERFECT HAND SIGNAL! Confidence: {hand_signal_detection.confidence:.1f}%")
                else:
                    print(f"❌ HAND SIGNAL VIOLATION! {hand_signal_detection.violation_type}")
        except Exception as e:
            print(f"⚠️ Hand signal detection error: {e}")
            import traceback
            traceback.print_exc()
        
        # Additional simple hand detection as fallback (using holistic model directly)
        if not detections['handSignal']['signal_detected']:
            try:
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                holistic_result = holistic.process(rgb_frame)
                
                # Simple check: if hands are detected and raised
                if holistic_result.left_hand_landmarks or holistic_result.right_hand_landmarks:
                    # Check if hand is raised (wrist above elbow or shoulder)
                    hand_raised = False
                    confidence = 0.0
                    
                    if holistic_result.pose_landmarks:
                        # Get pose landmarks
                        left_wrist = holistic_result.pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_WRIST]
                        right_wrist = holistic_result.pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_WRIST]
                        left_shoulder = holistic_result.pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER]
                        right_shoulder = holistic_result.pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER]
                        
                        # Stricter: wrist must be noticeably above shoulder (0.05 = ~5% of frame)
                        if holistic_result.left_hand_landmarks and left_wrist.y < left_shoulder.y - 0.05:
                            hand_raised = True
                            confidence = 65.0
                        elif holistic_result.right_hand_landmarks and right_wrist.y < right_shoulder.y - 0.05:
                            hand_raised = True
                            confidence = 65.0
                    
                    if hand_raised:
                        detections['handSignal']['signal_detected'] = True
                        detections['handSignal']['signal_type'] = 'raised_hand'
                        detections['handSignal']['compliance_validation'] = {
                            'compliance_status': 'DETECTED',
                            'violation_type': 'None',
                            'confidence': confidence,
                            'current_gps': None,
                            'nearest_station': None,
                            'distance_to_station': None,
                            'tolerance_meters': None
                        }
                        print(f"🤚 Simple hand signal detected at {timestamp_seconds:.1f}s (fallback method)")
            except Exception as e:
                pass  # Silently fail fallback detection
        
        # Enhanced Microsleep Detection with improved accuracy
        mesh_results = face_mesh.process(rgb_frame)
        if mesh_results.multi_face_landmarks:
            landmarks = mesh_results.multi_face_landmarks[0].landmark
            left_ear = self.eye_aspect_ratio(landmarks, self.LEFT_EYE_IDX, w, h)
            right_ear = self.eye_aspect_ratio(landmarks, self.RIGHT_EYE_IDX, w, h)
            avg_ear = (left_ear + right_ear) / 2.0
            
            # More sensitive microsleep detection with multiple thresholds
            if avg_ear < 0.15:  # Very strict threshold for clear drowsiness
                detections['microsleep']['detected'] = True
                detections['microsleep']['reason'] = 'severe_drowsiness'
                detections['microsleep']['ear_value'] = avg_ear
                print(f"😴 SEVERE MICROSLEEP! EAR: {avg_ear:.3f}")
            elif avg_ear < 0.18:  # Moderate threshold
                detections['microsleep']['detected'] = True
                detections['microsleep']['reason'] = 'drowsy_eyes'
                detections['microsleep']['ear_value'] = avg_ear
                print(f"😴 MICROSLEEP DETECTED! EAR: {avg_ear:.3f}")
            elif avg_ear < 0.22:  # Early warning threshold
                detections['microsleep']['detected'] = True
                detections['microsleep']['reason'] = 'fatigue_warning'
                detections['microsleep']['ear_value'] = avg_ear
                print(f"⚠️ FATIGUE WARNING! EAR: {avg_ear:.3f}")
        
        return detections

processor = VideoProcessor()

@app.route('/api/process-video', methods=['POST'])
def process_video():
    try:
        # Log which form file keys were received (so we can confirm Excel files are sent)
        form_file_keys = list(request.files.keys()) if request.files else []
        print(f"📎 POST /api/process-video — form file keys: {form_file_keys}")
        if 'video' not in request.files:
            return jsonify({'error': 'No video file provided'}), 400
        
        video_file = request.files['video']
        video_title = request.form.get('title', f'Video {datetime.now().strftime("%Y-%m-%d %H:%M")}')
        
        # Check file extension
        allowed_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.m4v', '.webm', '.flv', '.wmv']
        file_extension = os.path.splitext(video_file.filename.lower())[1]
        
        if file_extension not in allowed_extensions:
            supported_formats = ', '.join(allowed_extensions)
            return jsonify({'error': f'Unsupported video format: {file_extension}. Supported formats: {supported_formats}'}), 400
        
        # Save video temporarily with original extension
        video_id = str(uuid.uuid4())
        temp_path = f'/tmp/{video_id}{file_extension}'
        video_file.save(temp_path)
        
        # Process video
        cap = cv2.VideoCapture(temp_path)
        
        # Check if video opened successfully
        if not cap.isOpened():
            os.remove(temp_path)
            return jsonify({'error': f'Failed to open video file. The {file_extension} format may not be supported by OpenCV on this system.'}), 400
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        if fps <= 0 or total_frames <= 0:
            cap.release()
            os.remove(temp_path)
            return jsonify({'error': 'Invalid video file or corrupted video data.'}), 400
        
        duration_seconds = total_frames / fps
        
        # Load GPS early for OCR: use on-screen time (top-left) to match Excel times when video is a clip
        ocr_reference_date = None
        gps_entries_preloaded = None
        gps_log_file_early = request.files.get('gpsLog')
        if gps_log_file_early and gps_log_file_early.filename and gps_log_file_early.filename.lower().endswith(('.xlsx', '.xls')):
            try:
                gps_data_early = gps_log_file_early.read()
                gps_entries_preloaded = load_gps_log(gps_data_early)
                if gps_entries_preloaded and gps_entries_preloaded[0].get('real_datetime'):
                    ocr_reference_date = gps_entries_preloaded[0]['real_datetime']
                    print(f"🕐 OCR enabled: will match on-screen time to Excel (ref date {ocr_reference_date.date()})")
            except Exception as e:
                print(f"⚠️ Could not load GPS for OCR: {e}")
        
        detection_results = []
        frame_count = 0
        
        # Track which stations have been checked for signals
        station_signal_tracking = {}  # {station_id: {'expected_time': float, 'signal_detected': bool, 'detection_time': float}}
        
        # Process every 60th frame to reduce spam (was 15)
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # Process every 30th frame for better accuracy
            if frame_count % 30 == 0:
                timestamp_seconds = frame_count / fps
                detections = processor.process_frame(frame, timestamp_seconds)
                # OCR on-screen time (top-left) when GPS loaded - match video clip time to Excel
                if _OCR_AVAILABLE and ocr_reference_date:
                    try:
                        ocr_dt = extract_time_from_frame(frame, ocr_reference_date)
                        if ocr_dt is not None:
                            detections['ocr_timestamp'] = ocr_dt.isoformat()
                    except Exception:
                        pass
                
                # Add station information to every detection frame
                # Check if we're near any station that requires a signal
                if processor.perfect_hand_detector.current_gps:
                    nearest_rule, distance = processor.perfect_hand_detector.find_applicable_rule(
                        processor.perfect_hand_detector.current_gps
                    )
                    
                    if nearest_rule:
                        # Add station context to detection
                        detections['station_context'] = {
                            'station_id': nearest_rule.location_id,
                            'station_name': nearest_rule.station_name,
                            'distance_to_station': distance,
                            'within_tolerance': distance <= nearest_rule.tolerance_meters,
                            'signal_required': nearest_rule.signal_required,
                            'latitude': nearest_rule.latitude,
                            'longitude': nearest_rule.longitude,
                            'tolerance_meters': nearest_rule.tolerance_meters
                        }
                        
                        # Track if signal was required but not detected
                        # Only track if we're near the expected time window for this station
                        if nearest_rule.signal_required and distance <= nearest_rule.tolerance_meters:
                            station_key = nearest_rule.location_id
                            
                            # Get expected time for this station
                            expected_time = getattr(nearest_rule, 'expected_time', None)
                            tolerance_seconds = getattr(nearest_rule, 'tolerance_seconds', 10.0)
                            
                            # Only track if we're within the time window (expected_time ± tolerance)
                            within_time_window = True
                            if expected_time is not None and expected_time > 0:
                                time_diff = abs(timestamp_seconds - expected_time)
                                within_time_window = time_diff <= (tolerance_seconds * 2)  # Check within 2x tolerance window
                            
                            if within_time_window:
                                if station_key not in station_signal_tracking:
                                    station_signal_tracking[station_key] = {
                                        'station_name': nearest_rule.station_name,
                                        'expected_time': expected_time if expected_time else timestamp_seconds,
                                        'signal_detected': False,
                                        'detection_time': None,
                                        'distance': distance,
                                        'latitude': nearest_rule.latitude,
                                        'longitude': nearest_rule.longitude,
                                        'signal_required': True
                                    }
                                
                                # Update tracking if signal detected
                                if detections['handSignal']['signal_detected']:
                                    station_signal_tracking[station_key]['signal_detected'] = True
                                    station_signal_tracking[station_key]['detection_time'] = timestamp_seconds
                
                # Enhanced logging with all objects detected
                all_objects_str = ", ".join([f"{obj['class_name']}({obj['confidence']:.2f})" for obj in detections['objects']])
                if all_objects_str:
                    print(f"🔍 Frame {frame_count} ({timestamp_seconds:.1f}s): {all_objects_str}")
                
                detection_results.append(detections)
            
            frame_count += 1
        
        # Save video to a different location for playback
        video_filename = f"video_{video_id}{file_extension}"
        video_save_path = os.path.join('/tmp', video_filename)
        
        # Copy video before cleanup (different filename)
        import shutil
        shutil.copy2(temp_path, video_save_path)
        
        cap.release()
        os.remove(temp_path)  # Clean up original temp file
        phone_events = [d for d in detection_results if d['phone']['phone_detected']]
        hand_signal_events = [d for d in detection_results if d['handSignal']['signal_detected']]
        microsleep_events = [d for d in detection_results if d['microsleep']['detected']]
        bag_events = [d for d in detection_results if d['bags']['bags_detected']]
        
        # Generate Station Alert Compliance Report
        station_alerts = processor.station_alert_system.validate_station_compliance(
            detection_results, 
            processor.perfect_hand_detector.current_gps
        )
        station_report = processor.station_alert_system.generate_station_report(station_alerts)
        
        # Station Alerts validation: use uploaded Excel files OR disk files
        signal_validation = []
        signal_rules = []
        signal_coords_file = request.files.get('signalCoords')
        gps_log_file = request.files.get('gpsLog')
        sc_name = getattr(signal_coords_file, 'filename', None) if signal_coords_file else None
        gps_name = getattr(gps_log_file, 'filename', None) if gps_log_file else None
        print(f"📎 Station Excel files — signalCoords: {'present (' + (sc_name or 'no name') + ')' if sc_name else 'absent'} | gpsLog: {'present (' + (gps_name or 'no name') + ')' if gps_name else 'absent'}")
        if signal_coords_file and signal_coords_file.filename:
            print(f"📎 Received Signal Coords file: {signal_coords_file.filename} ({getattr(signal_coords_file, 'content_length', '?')} bytes)")
        if gps_log_file and gps_log_file.filename:
            print(f"📎 Received GPS Log file: {gps_log_file.filename}")

        try:
            if signal_coords_file and signal_coords_file.filename and signal_coords_file.filename.lower().endswith(('.xlsx', '.xls')):
                # Use uploaded Signal Coordinates file
                signal_data = signal_coords_file.read()
                print(f"📎 Signal Coords file size: {len(signal_data)} bytes")
                gdr_mas_rules = load_signal_rules_from_bytes(signal_data)
                print(f"📎 Loaded {len(gdr_mas_rules)} rules from Signal Coords Excel")
                if not gdr_mas_rules:
                    print("⚠️ No rules parsed from uploaded Signal Coords file - check column names (e.g. Latitude/Longitude or LATITUDE_RC/LONGITUDE_RC)")
                if gps_log_file and gps_log_file.filename and gps_log_file.filename.lower().endswith(('.xlsx', '.xls')):
                    # Use uploaded GPS log (GDR TO MAS) - correlate actual train times with signal locations
                    if gps_entries_preloaded:
                        gps_entries = gps_entries_preloaded
                    else:
                        gps_data = gps_log_file.read()
                        gps_entries = load_gps_log(gps_data)
                    print(f"📎 Loaded {len(gps_entries)} GPS log entries")
                    if gps_entries and gdr_mas_rules:
                        rules_with_time = correlate_sid_with_rtis(gdr_mas_rules, gps_entries, tolerance_seconds=3.0, top_n=5)
                        # Clip time windows to video duration
                        for r in rules_with_time:
                            tw = r.get('time_window', {})
                            r['time_window'] = {'start': max(0, tw.get('start', 0)), 'end': min(duration_seconds, tw.get('end', duration_seconds))}
                        print("📍 Using GPS-correlated times (GDR TO MAS + Signal Coordinates)")
                    else:
                        rules_with_time = assign_time_windows(gdr_mas_rules, duration_seconds, tolerance_seconds=3.0) if gdr_mas_rules else []
                else:
                    rules_with_time = assign_time_windows(gdr_mas_rules, duration_seconds, tolerance_seconds=3.0) if gdr_mas_rules else []
            else:
                # Fallback to disk files
                gdr_mas_rules = load_gdr_mas_rules()
                rules_with_time = assign_time_windows(gdr_mas_rules, duration_seconds, tolerance_seconds=3.0) if gdr_mas_rules else []
            
            if rules_with_time and duration_seconds > 0:
                signal_validation = validate_station_alerts(rules_with_time, detection_results)
                signal_rules = get_rules_for_frontend(rules_with_time)
                compliant_count = sum(1 for v in signal_validation if v['compliance'] == 'compliant')
                missed_count = sum(1 for v in signal_validation if v['compliance'] == 'missed')
                print(f"\n🚉 STATION ALERTS: Pilot Did It ✓ {compliant_count} | Pilot Missed It ✗ {missed_count} (from {len(signal_validation)} rules)")
        except Exception as e:
            print(f"⚠️ Station alerts loader: {e}")
            import traceback
            traceback.print_exc()
        
        print(f"\n🚉 STATION COMPLIANCE REPORT:")
        print(f"📊 Total Stations: {station_report['summary']['total_stations']}")
        print(f"✅ Compliant: {station_report['summary']['compliant']}")
        print(f"❌ Missed: {station_report['summary']['missed']}")
        print(f"⏰ Late: {station_report['summary']['late']}")
        print(f"⚡ Early: {station_report['summary']['early']}")
        print(f"📈 Compliance Rate: {station_report['summary']['compliance_rate']:.1f}%")
        
        # Format duration
        minutes = int(duration_seconds // 60)
        seconds = int(duration_seconds % 60)
        duration_str = f"{minutes}m {seconds}s"
        
        # Create detailed events for frontend
        events = []
        for i, detection in enumerate(detection_results):
            timestamp = detection['timestamp_seconds']
            time_str = f"{int(timestamp//3600):02d}:{int((timestamp%3600)//60):02d}:{int(timestamp%60):02d}"
            
            if detection['phone']['phone_detected']:
                events.append({
                    'time': time_str,
                    'timestamp': timestamp,  # Add timestamp in seconds for video seeking
                    'type': 'phone_usage',
                    'details': f"Confidence: {int(detection['phone']['detections'][0]['confidence']*100)}%",
                    'status': 'violation'
                })
            
            if detection['handSignal']['signal_detected']:
                compliance_validation = detection['handSignal'].get('compliance_validation', {})
                is_violation = compliance_validation.get('compliance_status') == 'VIOLATION'
                station_context = detection.get('station_context', {})
                
                events.append({
                    'time': time_str,
                    'timestamp': timestamp,  # Add timestamp in seconds for video seeking
                    'type': 'hand_signal',
                    'details': f"Signal: {detection['handSignal']['signal_type']} - {compliance_validation.get('violation_type', 'Compliant signal')}",
                    'status': 'violation' if is_violation else 'compliant',
                    'compliance_info': compliance_validation,
                    'confidence': compliance_validation.get('confidence', 0),
                    'station': station_context.get('station_name') or compliance_validation.get('nearest_station', 'Unknown'),
                    'station_id': station_context.get('station_id'),
                    'distance_to_station': station_context.get('distance_to_station') or compliance_validation.get('distance_to_station'),
                    'gps_location': station_context.get('latitude') and station_context.get('longitude') and {
                        'latitude': station_context['latitude'],
                        'longitude': station_context['longitude']
                    } or None
                })
            
            # Check for missed signals - if we're at a station that requires a signal but none was detected
            station_context = detection.get('station_context', {})
            if station_context.get('signal_required') and station_context.get('within_tolerance'):
                if not detection['handSignal']['signal_detected']:
                    # This is a potential missed signal - add to events
                    events.append({
                        'time': time_str,
                        'timestamp': timestamp,  # Add timestamp in seconds for video seeking
                        'type': 'missed_signal',
                        'details': f"MISSED SIGNAL - Required at {station_context.get('station_name')} but no signal detected",
                        'status': 'missed',
                        'station': station_context.get('station_name', 'Unknown'),
                        'station_id': station_context.get('station_id'),
                        'distance_to_station': station_context.get('distance_to_station'),
                        'gps_location': {
                            'latitude': station_context.get('latitude'),
                            'longitude': station_context.get('longitude')
                        } if station_context.get('latitude') else None,
                        'compliance_info': {
                            'compliance_status': 'MISSED',
                            'violation_type': f"MISSED SIGNAL - Required at {station_context.get('station_name')}",
                            'confidence': 0
                        }
                    })
            
            if detection['microsleep']['detected']:
                severity = detection['microsleep']['reason']
                ear_value = detection['microsleep'].get('ear_value', 0)
                
                # Determine severity level for display
                if severity == 'severe_drowsiness':
                    severity_text = "CRITICAL - Severe Drowsiness"
                    status = 'violation'
                elif severity == 'drowsy_eyes':
                    severity_text = "HIGH - Drowsy Eyes Detected"
                    status = 'violation'
                elif severity == 'fatigue_warning':
                    severity_text = "MEDIUM - Fatigue Warning"
                    status = 'violation'
                else:
                    severity_text = f"Reason: {severity}"
                    status = 'violation'
                
                events.append({
                    'time': time_str,
                    'timestamp': timestamp,  # Add timestamp in seconds for video seeking
                    'type': 'microsleep',
                    'details': f"{severity_text} (EAR: {ear_value:.3f})",
                    'status': status,
                    'severity': severity,
                    'ear_value': ear_value
                })
            
            if detection['bags']['bags_detected']:
                events.append({
                    'time': time_str,
                    'timestamp': timestamp,  # Add timestamp in seconds for video seeking
                    'type': 'bag_detection',
                    'details': f"Bags: {len(detection['bags']['detections'])}",
                    'status': 'compliant'
                })
        
        # Add missed signals from station tracking
        for station_key, tracking in station_signal_tracking.items():
            if tracking['signal_required'] and not tracking['signal_detected']:
                expected_time = tracking['expected_time']
                time_str = f"{int(expected_time//3600):02d}:{int((expected_time%3600)//60):02d}:{int(expected_time%60):02d}"
                events.append({
                    'time': time_str,
                    'timestamp': expected_time,  # Add timestamp in seconds for video seeking
                    'type': 'missed_signal',
                    'details': f"MISSED SIGNAL - Required at {tracking['station_name']} but no signal detected",
                    'status': 'missed',
                    'station': tracking['station_name'],
                    'station_id': station_key,
                    'distance_to_station': tracking['distance'],
                    'gps_location': {
                        'latitude': tracking['latitude'],
                        'longitude': tracking['longitude']
                    },
                    'compliance_info': {
                        'compliance_status': 'MISSED',
                        'violation_type': f"MISSED SIGNAL - Required at {tracking['station_name']}",
                        'confidence': 0
                    }
                })
        
        # Count violations including perfect hand signal violations
        hand_signal_violations = sum(1 for d in detection_results 
                                   if d['handSignal']['signal_detected'] and 
                                   d['handSignal'].get('compliance_validation', {}).get('compliance_status') == 'VIOLATION')
        
        # Count missed signals
        missed_signals_count = len([e for e in events if e['type'] == 'missed_signal'])
        
        video_data = {
            'id': video_id,
            'title': video_title,
            'date': datetime.now().isoformat()[:10],
            'status': 'completed',
            'duration': duration_str,
            'videoUrl': f'http://localhost:9001/video/{video_filename}',  # Video URL
            'phoneUsage': len(phone_events),
            'handSignals': len(hand_signal_events),
            'bagsDetected': len(bag_events),
            'alerts': len(phone_events) + len(microsleep_events) + hand_signal_violations,
            'detectionResults': detection_results,
            'events': events,
            'thumbnail': 'https://images.unsplash.com/photo-1474487548417-781cb71495f3?ixlib=rb-1.2.1&auto=format&fit=crop&q=80&w=2884&h=1618',
            'detectedObjects': {},
            'handSignalViolations': hand_signal_violations,
            'missedSignals': missed_signals_count,
            'stationAlerts': station_report,  # Add station compliance data
            'signalValidation': signal_validation,  # Requirement team GDR-MAS validation
            'signalRules': signal_rules,  # Rules for frontend display
            'signalRulesUploaded': bool(signal_coords_file and signal_coords_file.filename),  # True if user uploaded rules file
            'signalRulesParseError': 'Could not parse rules Excel. Use columns: Latitude, Longitude (or LATITUDE_RC, LONGITUDE_RC), Station, SIG_CODE.' if (signal_coords_file and signal_coords_file.filename and not signal_rules) else None,
            'handSignalCompliance': {
                'total_signals_detected': len(hand_signal_events),
                'compliant_signals': len([e for e in events if e['type'] == 'hand_signal' and e['status'] == 'compliant']),
                'violation_signals': hand_signal_violations,
                'missed_signals': missed_signals_count,
                'compliance_rate': (len([e for e in events if e['type'] == 'hand_signal' and e['status'] == 'compliant']) / 
                                  max(1, len(hand_signal_events) + missed_signals_count) * 100) if (len(hand_signal_events) + missed_signals_count) > 0 else 100.0
            }
        }
        
        # Debug: Print video URL and check if file exists
        print(f"\n🎥 VIDEO SERVING INFO:")
        print(f"Video ID: {video_id}")
        print(f"Video filename: {video_filename}")
        print(f"Video save path: {video_save_path}")
        print(f"Video URL: {video_data['videoUrl']}")
        print(f"File exists: {os.path.exists(video_save_path)}")
        if os.path.exists(video_save_path):
            print(f"File size: {os.path.getsize(video_save_path)} bytes")
        
        # Add detected objects summary
        all_objects = {}
        for result in detection_results:
            for obj in result['objects']:
                class_name = obj['class_name']
                if class_name in all_objects:
                    all_objects[class_name] += 1
                else:
                    all_objects[class_name] = 1
        
        video_data['detectedObjects'] = all_objects
        
        # If no specific detections but objects found, create events for interesting objects
        if len(events) == 0 and all_objects:
            print(f"No specific detections, but found objects: {all_objects}")
            # Create events for interesting objects like phones, laptops, etc.
            for result in detection_results:
                timestamp = result['timestamp_seconds']
                time_str = f"{int(timestamp//3600):02d}:{int((timestamp%3600)//60):02d}:{int(timestamp%60):02d}"
                
                for obj in result['objects']:
                    # Only show interesting objects
                    if obj['class_name'] in ['cell phone', 'laptop', 'backpack', 'handbag', 'suitcase']:
                        events.append({
                            'time': time_str,
                            'type': 'object_detection',
                            'details': f"Detected: {obj['class_name']} ({int(obj['confidence']*100)}%)",
                            'status': 'compliant'
                        })
        
        return jsonify({
            'success': True,
            'video': video_data,
            'message': f'Video processed successfully. Found {len(phone_events)} phone usage events, {len(hand_signal_events)} hand signals, {len(microsleep_events)} microsleep events, {len(bag_events)} bag detections.',
            'detection_summary': {
                'phone_violations': len(phone_events),
                'hand_signals': len(hand_signal_events),
                'hand_signal_violations': hand_signal_violations,
                'microsleep_events': len(microsleep_events),
                'bag_detections': len(bag_events),
                'total_objects': sum(len(d['objects']) for d in detection_results)
            }
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/set-gps', methods=['POST'])
def set_gps_location():
    """Set GPS coordinates for perfect hand signal validation"""
    try:
        data = request.get_json()
        latitude = float(data.get('latitude', 0))
        longitude = float(data.get('longitude', 0))
        
        processor.perfect_hand_detector.set_current_gps(latitude, longitude)
        
        return jsonify({
            'success': True,
            'message': f'GPS location set to {latitude}, {longitude}',
            'gps_location': {'latitude': latitude, 'longitude': longitude}
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/valid-locations', methods=['GET'])
def get_valid_locations():
    """Get all valid signal locations from Excel file"""
    try:
        locations = [{
            'index': rule.location_id,
            'latitude': rule.latitude,
            'longitude': rule.longitude,
            'station_name': rule.station_name,
            'tolerance_meters': rule.tolerance_meters,
            'mandatory': rule.mandatory
        } for rule in processor.perfect_hand_detector.signal_rules]
        
        return jsonify({
            'success': True,
            'locations': locations,
            'total_count': len(locations)
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/api/test-video/<filename>', methods=['GET'])
def test_video_access(filename):
    """Test endpoint to check video file access"""
    try:
        video_path = os.path.join('/tmp', filename)
        exists = os.path.exists(video_path)
        size = os.path.getsize(video_path) if exists else 0
        
        return jsonify({
            'filename': filename,
            'path': video_path,
            'exists': exists,
            'size': size,
            'url': f'http://localhost:9001/video/{filename}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/video/<filename>', methods=['GET', 'OPTIONS'])
def serve_video_tmp(filename):
    if request.method == 'OPTIONS':
        response = jsonify({'status': 'ok'})
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        return response
    
    try:
        video_path = os.path.join('/tmp', filename)
        if not os.path.exists(video_path):
            return jsonify({'error': 'Video file not found'}), 404
        
        # Get file extension to determine MIME type
        file_ext = os.path.splitext(filename)[1].lower()
        mime_types = {
            '.mp4': 'video/mp4',
            '.m4v': 'video/mp4',
            '.avi': 'video/x-msvideo',
            '.mov': 'video/quicktime',
            '.mkv': 'video/x-matroska',
            '.webm': 'video/webm',
            '.flv': 'video/x-flv',
            '.wmv': 'video/x-ms-wmv'
        }
        
        mime_type = mime_types.get(file_ext, 'video/mp4')
        
        response = send_file(video_path, mimetype=mime_type)
        response.headers['Access-Control-Allow-Origin'] = '*'
        response.headers['Access-Control-Allow-Methods'] = 'GET, OPTIONS'
        response.headers['Access-Control-Allow-Headers'] = 'Content-Type'
        response.headers['Cache-Control'] = 'no-cache'
        
        return response
    except Exception as e:
        return jsonify({'error': f'Error serving video: {str(e)}'}), 500

@app.route('/', methods=['GET'])
def home():
    return jsonify({'message': 'Video Processing API is running', 'status': 'healthy'})

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 9001))
    app.run(host='0.0.0.0', port=port, debug=False)