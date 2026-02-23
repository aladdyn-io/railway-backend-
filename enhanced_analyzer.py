import os
import cv2
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime, timedelta
import json
import sqlite3
from pathlib import Path
import logging
from mediapipe import solutions as mp_solutions
from ultralytics import YOLO
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

@dataclass
class AnalysisConfig:
    """Enhanced configuration for analysis"""
    # Detection thresholds
    pose_confidence: float = 0.7
    hand_confidence: float = 0.7
    face_confidence: float = 0.8
    object_confidence: float = 0.6
    
    # Analysis parameters
    smoothing_frames: int = 3
    cooldown_seconds: float = 1.5
    max_tracking_objects: int = 20
    
    # Quality settings
    min_detection_quality: float = 0.6
    enable_tracking: bool = True
    save_analytics: bool = True
    
    # Output settings
    output_folder: str = "enhanced_analysis"
    enable_visualization: bool = True
    generate_reports: bool = True

@dataclass
class DetectionEvent:
    """Enhanced detection event with metadata"""
    timestamp: float
    event_type: str
    confidence: float
    bbox: Optional[Tuple[int, int, int, int]] = None
    landmarks: Optional[Dict] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    frame_path: Optional[str] = None
    quality_score: float = 0.0

class EnhancedAnalyzer:
    """Professional-grade analysis engine with advanced detection and analytics"""
    
    def __init__(self, config: AnalysisConfig):
        self.config = config
        self.setup_logging()
        self.initialize_models()
        self.setup_database()
        self.detection_history = []
        self.performance_metrics = {}
        
    def setup_logging(self):
        """Setup comprehensive logging"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(f'{self.config.output_folder}/analysis.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
    def initialize_models(self):
        """Initialize all AI models with optimized settings"""
        try:
            # MediaPipe models
            self.holistic = mp_solutions.holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,  # Higher complexity for better accuracy
                min_detection_confidence=self.config.pose_confidence,
                min_tracking_confidence=self.config.hand_confidence,
                enable_segmentation=True,
                refine_face_landmarks=True
            )
            
            self.face_mesh = mp_solutions.face_mesh.FaceMesh(
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=self.config.face_confidence,
                min_tracking_confidence=self.config.face_confidence
            )
            
            # YOLO model for object detection
            self.yolo = YOLO('yolov8x.pt')  # Use larger model for better accuracy
            
            # Define detection classes
            self.PHONE_CLASSES = [67]  # cell phone
            self.BAG_CLASSES = [24, 25, 26, 27]  # backpack, handbag, suitcase, briefcase
            self.PERSON_CLASS = 0
            
            self.logger.info("All models initialized successfully")
            
        except Exception as e:
            self.logger.error(f"Model initialization failed: {e}")
            raise
            
    def setup_database(self):
        """Setup SQLite database for storing analysis results"""
        os.makedirs(self.config.output_folder, exist_ok=True)
        self.db_path = os.path.join(self.config.output_folder, 'analysis_results.db')
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Create tables
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS detections (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                timestamp REAL,
                event_type TEXT,
                confidence REAL,
                bbox_x INTEGER,
                bbox_y INTEGER,
                bbox_w INTEGER,
                bbox_h INTEGER,
                quality_score REAL,
                metadata TEXT,
                frame_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS analytics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                video_path TEXT,
                total_duration REAL,
                total_detections INTEGER,
                accuracy_score REAL,
                processing_time REAL,
                model_versions TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def calculate_quality_score(self, detection_data: Dict) -> float:
        """Calculate quality score for detection"""
        score = 0.0
        factors = 0
        
        # Confidence factor
        if 'confidence' in detection_data:
            score += detection_data['confidence'] * 0.4
            factors += 0.4
            
        # Visibility factor (for pose landmarks)
        if 'visibility' in detection_data:
            score += detection_data['visibility'] * 0.3
            factors += 0.3
            
        # Size factor (larger detections are generally more reliable)
        if 'bbox' in detection_data:
            bbox = detection_data['bbox']
            area = (bbox[2] - bbox[0]) * (bbox[3] - bbox[1])
            normalized_area = min(area / (640 * 480), 1.0)  # Normalize to standard resolution
            score += normalized_area * 0.3
            factors += 0.3
            
        return score / factors if factors > 0 else 0.0
        
    def advanced_gesture_detection(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Enhanced gesture detection with confidence scoring"""
        events = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w = frame.shape[:2]
        
        # Holistic detection
        holistic_results = self.holistic.process(rgb_frame)
        
        if holistic_results.pose_landmarks:
            # Enhanced hand raise detection
            hand_events = self._detect_hand_gestures(
                holistic_results.pose_landmarks,
                holistic_results.left_hand_landmarks,
                holistic_results.right_hand_landmarks,
                timestamp, w, h
            )
            events.extend(hand_events)
            
        return events
        
    def _detect_hand_gestures(self, pose_landmarks, left_hand, right_hand, timestamp, w, h) -> List[DetectionEvent]:
        """Advanced hand gesture detection with multiple criteria"""
        events = []
        
        try:
            # Get key pose landmarks
            left_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            nose = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.NOSE]
            
            # Check left hand
            if left_hand:
                confidence = self._calculate_hand_raise_confidence(
                    left_hand, pose_landmarks, 'left'
                )
                if confidence > self.config.min_detection_quality:
                    # Calculate bounding box
                    hand_points = [(lm.x * w, lm.y * h) for lm in left_hand.landmark]
                    x_coords, y_coords = zip(*hand_points)
                    bbox = (int(min(x_coords)), int(min(y_coords)), 
                           int(max(x_coords)), int(max(y_coords)))
                    
                    event = DetectionEvent(
                        timestamp=timestamp,
                        event_type="hand_raised_left",
                        confidence=confidence,
                        bbox=bbox,
                        quality_score=self.calculate_quality_score({
                            'confidence': confidence,
                            'visibility': left_shoulder.visibility,
                            'bbox': bbox
                        }),
                        metadata={
                            'hand_side': 'left',
                            'gesture_type': 'raised_hand',
                            'shoulder_visibility': left_shoulder.visibility
                        }
                    )
                    events.append(event)
                    
            # Check right hand (similar logic)
            if right_hand:
                confidence = self._calculate_hand_raise_confidence(
                    right_hand, pose_landmarks, 'right'
                )
                if confidence > self.config.min_detection_quality:
                    hand_points = [(lm.x * w, lm.y * h) for lm in right_hand.landmark]
                    x_coords, y_coords = zip(*hand_points)
                    bbox = (int(min(x_coords)), int(min(y_coords)), 
                           int(max(x_coords)), int(max(y_coords)))
                    
                    event = DetectionEvent(
                        timestamp=timestamp,
                        event_type="hand_raised_right",
                        confidence=confidence,
                        bbox=bbox,
                        quality_score=self.calculate_quality_score({
                            'confidence': confidence,
                            'visibility': right_shoulder.visibility,
                            'bbox': bbox
                        }),
                        metadata={
                            'hand_side': 'right',
                            'gesture_type': 'raised_hand',
                            'shoulder_visibility': right_shoulder.visibility
                        }
                    )
                    events.append(event)
                    
        except Exception as e:
            self.logger.warning(f"Hand gesture detection error: {e}")
            
        return events
        
    def _calculate_hand_raise_confidence(self, hand_landmarks, pose_landmarks, side: str) -> float:
        """Calculate confidence score for hand raise detection"""
        try:
            # Get relevant landmarks
            wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
            index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            
            shoulder_idx = (mp_solutions.pose.PoseLandmark.LEFT_SHOULDER if side == 'left' 
                          else mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER)
            shoulder = pose_landmarks.landmark[shoulder_idx]
            
            # Height criterion (hand above shoulder)
            height_score = max(0, (shoulder.y - wrist.y) * 2)  # Normalize and amplify
            
            # Extension criterion (fingers extended)
            extension_score = self._calculate_finger_extension(hand_landmarks)
            
            # Visibility criterion
            visibility_score = min(wrist.visibility, shoulder.visibility)
            
            # Combined confidence
            confidence = (height_score * 0.4 + extension_score * 0.4 + visibility_score * 0.2)
            return min(confidence, 1.0)
            
        except Exception:
            return 0.0
            
    def _calculate_finger_extension(self, hand_landmarks) -> float:
        """Calculate how extended the fingers are"""
        try:
            # Check key finger landmarks
            wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
            index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            
            # Calculate distances from wrist
            index_dist = np.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
            middle_dist = np.sqrt((middle_tip.x - wrist.x)**2 + (middle_tip.y - wrist.y)**2)
            
            # Normalize and combine
            avg_extension = (index_dist + middle_dist) / 2
            return min(avg_extension * 5, 1.0)  # Scale appropriately
            
        except Exception:
            return 0.0
            
    def advanced_object_detection(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Enhanced object detection with tracking and quality assessment"""
        events = []
        
        # YOLO detection
        results = self.yolo.predict(frame, verbose=False, conf=self.config.object_confidence)[0]
        
        for box in results.boxes:
            cls_id = int(box.cls[0])
            confidence = float(box.conf[0])
            bbox = box.xyxy[0].cpu().numpy().astype(int)
            
            # Phone detection
            if cls_id in self.PHONE_CLASSES:
                quality_score = self.calculate_quality_score({
                    'confidence': confidence,
                    'bbox': bbox
                })
                
                if quality_score > self.config.min_detection_quality:
                    event = DetectionEvent(
                        timestamp=timestamp,
                        event_type="mobile_phone_detected",
                        confidence=confidence,
                        bbox=tuple(bbox),
                        quality_score=quality_score,
                        metadata={
                            'object_class': 'mobile_phone',
                            'yolo_class_id': cls_id
                        }
                    )
                    events.append(event)
                    
            # Bag detection
            elif cls_id in self.BAG_CLASSES:
                quality_score = self.calculate_quality_score({
                    'confidence': confidence,
                    'bbox': bbox
                })
                
                if quality_score > self.config.min_detection_quality:
                    event = DetectionEvent(
                        timestamp=timestamp,
                        event_type="bag_detected",
                        confidence=confidence,
                        bbox=tuple(bbox),
                        quality_score=quality_score,
                        metadata={
                            'object_class': 'bag',
                            'yolo_class_id': cls_id
                        }
                    )
                    events.append(event)
                    
        return events
        
    def advanced_drowsiness_detection(self, frame: np.ndarray, timestamp: float) -> List[DetectionEvent]:
        """Enhanced drowsiness detection with multiple indicators"""
        events = []
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Face mesh detection
        face_results = self.face_mesh.process(rgb_frame)
        
        if face_results.multi_face_landmarks:
            for face_landmarks in face_results.multi_face_landmarks:
                # Calculate multiple drowsiness indicators
                ear_score = self._calculate_eye_aspect_ratio(face_landmarks, frame.shape)
                head_pose_score = self._calculate_head_pose_drowsiness(face_landmarks, frame.shape)
                blink_rate_score = self._calculate_blink_pattern(face_landmarks, timestamp)
                
                # Combined drowsiness confidence
                drowsiness_confidence = (ear_score * 0.4 + head_pose_score * 0.4 + blink_rate_score * 0.2)
                
                if drowsiness_confidence > self.config.min_detection_quality:
                    # Calculate face bounding box
                    h, w = frame.shape[:2]
                    face_points = [(lm.x * w, lm.y * h) for lm in face_landmarks.landmark]
                    x_coords, y_coords = zip(*face_points)
                    bbox = (int(min(x_coords)), int(min(y_coords)), 
                           int(max(x_coords)), int(max(y_coords)))
                    
                    event = DetectionEvent(
                        timestamp=timestamp,
                        event_type="drowsiness_detected",
                        confidence=drowsiness_confidence,
                        bbox=bbox,
                        quality_score=self.calculate_quality_score({
                            'confidence': drowsiness_confidence,
                            'bbox': bbox
                        }),
                        metadata={
                            'ear_score': ear_score,
                            'head_pose_score': head_pose_score,
                            'blink_rate_score': blink_rate_score,
                            'detection_method': 'multi_indicator'
                        }
                    )
                    events.append(event)
                    
        return events
        
    def _calculate_eye_aspect_ratio(self, face_landmarks, frame_shape) -> float:
        """Calculate eye aspect ratio for drowsiness detection"""
        try:
            h, w = frame_shape[:2]
            
            # Left eye landmarks
            left_eye_indices = [33, 160, 158, 133, 153, 144]
            left_eye_points = np.array([(face_landmarks.landmark[i].x * w, 
                                       face_landmarks.landmark[i].y * h) for i in left_eye_indices])
            
            # Right eye landmarks  
            right_eye_indices = [362, 385, 387, 263, 373, 380]
            right_eye_points = np.array([(face_landmarks.landmark[i].x * w, 
                                        face_landmarks.landmark[i].y * h) for i in right_eye_indices])
            
            # Calculate EAR for both eyes
            left_ear = self._compute_ear(left_eye_points)
            right_ear = self._compute_ear(right_eye_points)
            
            avg_ear = (left_ear + right_ear) / 2.0
            
            # Convert to drowsiness score (lower EAR = higher drowsiness)
            ear_threshold = 0.25
            if avg_ear < ear_threshold:
                return (ear_threshold - avg_ear) / ear_threshold
            return 0.0
            
        except Exception:
            return 0.0
            
    def _compute_ear(self, eye_points) -> float:
        """Compute eye aspect ratio from eye landmarks"""
        try:
            # Vertical distances
            A = np.linalg.norm(eye_points[1] - eye_points[5])
            B = np.linalg.norm(eye_points[2] - eye_points[4])
            
            # Horizontal distance
            C = np.linalg.norm(eye_points[0] - eye_points[3])
            
            # EAR formula
            ear = (A + B) / (2.0 * C)
            return ear
            
        except Exception:
            return 0.3  # Default normal EAR
            
    def _calculate_head_pose_drowsiness(self, face_landmarks, frame_shape) -> float:
        """Calculate head pose indicators of drowsiness"""
        # Simplified head pose calculation
        # In a full implementation, this would use 3D pose estimation
        try:
            h, w = frame_shape[:2]
            
            # Get nose and chin landmarks
            nose = face_landmarks.landmark[1]
            chin = face_landmarks.landmark[175]
            
            # Calculate head tilt (simplified)
            nose_y = nose.y * h
            chin_y = chin.y * h
            
            # Head dropping indicator
            head_drop_ratio = (chin_y - nose_y) / h
            
            # Convert to drowsiness score
            if head_drop_ratio > 0.15:  # Threshold for head dropping
                return min((head_drop_ratio - 0.15) * 5, 1.0)
            return 0.0
            
        except Exception:
            return 0.0
            
    def _calculate_blink_pattern(self, face_landmarks, timestamp: float) -> float:
        """Analyze blink patterns for drowsiness (simplified)"""
        # This would require temporal analysis across frames
        # For now, return a placeholder
        return 0.0
        
    def process_video_enhanced(self, video_path: str) -> Dict[str, Any]:
        """Enhanced video processing with comprehensive analysis"""
        start_time = datetime.now()
        self.logger.info(f"Starting enhanced analysis of {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open video: {video_path}")
            
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        
        all_events = []
        frame_count = 0
        
        # Processing progress tracking
        progress_interval = max(1, total_frames // 100)
        
        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                    
                timestamp = frame_count / fps
                
                # Run all detection modules
                gesture_events = self.advanced_gesture_detection(frame, timestamp)
                object_events = self.advanced_object_detection(frame, timestamp)
                drowsiness_events = self.advanced_drowsiness_detection(frame, timestamp)
                
                # Combine all events
                frame_events = gesture_events + object_events + drowsiness_events
                
                # Save high-quality detections
                for event in frame_events:
                    if event.quality_score > self.config.min_detection_quality:
                        # Save frame if configured
                        if self.config.enable_visualization:
                            event.frame_path = self._save_detection_frame(frame, event, timestamp)
                        
                        all_events.append(event)
                        self._store_detection_in_db(event)
                
                frame_count += 1
                
                # Progress logging
                if frame_count % progress_interval == 0:
                    progress = (frame_count / total_frames) * 100
                    self.logger.info(f"Processing progress: {progress:.1f}%")
                    
        finally:
            cap.release()
            
        # Calculate performance metrics
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Generate comprehensive analysis report
        analysis_results = self._generate_analysis_report(
            all_events, video_path, duration, processing_time
        )
        
        # Store analytics in database
        self._store_analytics_in_db(video_path, duration, len(all_events), processing_time)
        
        self.logger.info(f"Analysis completed in {processing_time:.2f} seconds")
        return analysis_results
        
    def _save_detection_frame(self, frame: np.ndarray, event: DetectionEvent, timestamp: float) -> str:
        """Save frame with detection annotations"""
        try:
            # Create annotated frame
            annotated_frame = frame.copy()
            
            # Draw bounding box if available
            if event.bbox:
                x1, y1, x2, y2 = event.bbox
                cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Add label
                label = f"{event.event_type} ({event.confidence:.2f})"
                cv2.putText(annotated_frame, label, (x1, y1-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            
            # Save frame
            filename = f"{event.event_type}_{int(timestamp*1000)}.jpg"
            filepath = os.path.join(self.config.output_folder, "frames", filename)
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            cv2.imwrite(filepath, annotated_frame)
            return filepath
            
        except Exception as e:
            self.logger.error(f"Error saving detection frame: {e}")
            return ""
            
    def _store_detection_in_db(self, event: DetectionEvent):
        """Store detection event in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            bbox_data = event.bbox if event.bbox else (None, None, None, None)
            
            cursor.execute('''
                INSERT INTO detections 
                (timestamp, event_type, confidence, bbox_x, bbox_y, bbox_w, bbox_h, 
                 quality_score, metadata, frame_path)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                event.timestamp, event.event_type, event.confidence,
                bbox_data[0], bbox_data[1], 
                bbox_data[2] - bbox_data[0] if bbox_data[0] else None,
                bbox_data[3] - bbox_data[1] if bbox_data[1] else None,
                event.quality_score, json.dumps(event.metadata), event.frame_path
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Database storage error: {e}")
            
    def _store_analytics_in_db(self, video_path: str, duration: float, 
                              total_detections: int, processing_time: float):
        """Store analytics summary in database"""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            model_versions = {
                'mediapipe': 'latest',
                'yolo': 'v8x',
                'analyzer': '2.0'
            }
            
            cursor.execute('''
                INSERT INTO analytics 
                (video_path, total_duration, total_detections, processing_time, model_versions)
                VALUES (?, ?, ?, ?, ?)
            ''', (
                video_path, duration, total_detections, processing_time,
                json.dumps(model_versions)
            ))
            
            conn.commit()
            conn.close()
            
        except Exception as e:
            self.logger.error(f"Analytics storage error: {e}")
            
    def _generate_analysis_report(self, events: List[DetectionEvent], 
                                 video_path: str, duration: float, 
                                 processing_time: float) -> Dict[str, Any]:
        """Generate comprehensive analysis report"""
        
        # Group events by type
        events_by_type = {}
        for event in events:
            if event.event_type not in events_by_type:
                events_by_type[event.event_type] = []
            events_by_type[event.event_type].append(event)
            
        # Calculate statistics
        stats = {
            'total_events': len(events),
            'events_by_type': {k: len(v) for k, v in events_by_type.items()},
            'average_confidence': np.mean([e.confidence for e in events]) if events else 0,
            'average_quality': np.mean([e.quality_score for e in events]) if events else 0,
            'video_duration': duration,
            'processing_time': processing_time,
            'processing_speed': duration / processing_time if processing_time > 0 else 0,
            'events_per_minute': (len(events) / duration) * 60 if duration > 0 else 0
        }
        
        # Generate timeline data
        timeline = []
        for event in sorted(events, key=lambda x: x.timestamp):
            timeline.append({
                'timestamp': event.timestamp,
                'time_formatted': f"{int(event.timestamp//60):02d}:{int(event.timestamp%60):02d}",
                'event_type': event.event_type,
                'confidence': event.confidence,
                'quality_score': event.quality_score,
                'frame_path': event.frame_path,
                'metadata': event.metadata
            })
            
        # Performance analysis
        performance = {
            'fps_processed': len(events) / processing_time if processing_time > 0 else 0,
            'memory_efficient': True,  # Placeholder
            'model_accuracy': self._estimate_model_accuracy(events),
            'detection_density': len(events) / duration if duration > 0 else 0
        }
        
        return {
            'video_info': {
                'path': video_path,
                'duration': duration,
                'processed_at': datetime.now().isoformat()
            },
            'statistics': stats,
            'timeline': timeline,
            'events_by_type': events_by_type,
            'performance': performance,
            'quality_metrics': self._calculate_quality_metrics(events)
        }
        
    def _estimate_model_accuracy(self, events: List[DetectionEvent]) -> float:
        """Estimate overall model accuracy based on confidence and quality scores"""
        if not events:
            return 0.0
            
        # Weighted average of confidence and quality scores
        total_score = sum(e.confidence * 0.6 + e.quality_score * 0.4 for e in events)
        return total_score / len(events)
        
    def _calculate_quality_metrics(self, events: List[DetectionEvent]) -> Dict[str, float]:
        """Calculate various quality metrics for the analysis"""
        if not events:
            return {}
            
        confidences = [e.confidence for e in events]
        quality_scores = [e.quality_score for e in events]
        
        return {
            'confidence_mean': np.mean(confidences),
            'confidence_std': np.std(confidences),
            'confidence_min': np.min(confidences),
            'confidence_max': np.max(confidences),
            'quality_mean': np.mean(quality_scores),
            'quality_std': np.std(quality_scores),
            'high_quality_ratio': sum(1 for q in quality_scores if q > 0.8) / len(quality_scores)
        }
        
    def generate_visualization_report(self, analysis_results: Dict[str, Any]) -> str:
        """Generate visual analysis report with charts and graphs"""
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
            
            # Set style
            plt.style.use('seaborn-v0_8')
            fig, axes = plt.subplots(2, 2, figsize=(15, 10))
            fig.suptitle('Enhanced Video Analysis Report', fontsize=16, fontweight='bold')
            
            # 1. Events by type (bar chart)
            events_by_type = analysis_results['statistics']['events_by_type']
            if events_by_type:
                axes[0, 0].bar(events_by_type.keys(), events_by_type.values())
                axes[0, 0].set_title('Detections by Type')
                axes[0, 0].tick_params(axis='x', rotation=45)
            
            # 2. Timeline (scatter plot)
            timeline = analysis_results['timeline']
            if timeline:
                timestamps = [t['timestamp'] for t in timeline]
                confidences = [t['confidence'] for t in timeline]
                axes[0, 1].scatter(timestamps, confidences, alpha=0.6)
                axes[0, 1].set_title('Detection Timeline')
                axes[0, 1].set_xlabel('Time (seconds)')
                axes[0, 1].set_ylabel('Confidence')
            
            # 3. Quality distribution (histogram)
            quality_scores = [t['quality_score'] for t in timeline]
            if quality_scores:
                axes[1, 0].hist(quality_scores, bins=20, alpha=0.7)
                axes[1, 0].set_title('Quality Score Distribution')
                axes[1, 0].set_xlabel('Quality Score')
                axes[1, 0].set_ylabel('Frequency')
            
            # 4. Performance metrics (text summary)
            perf = analysis_results['performance']
            stats_text = f"""
            Processing Speed: {perf['fps_processed']:.1f} FPS
            Model Accuracy: {perf['model_accuracy']:.2f}
            Detection Density: {perf['detection_density']:.2f}/sec
            Total Events: {analysis_results['statistics']['total_events']}
            """
            axes[1, 1].text(0.1, 0.5, stats_text, fontsize=12, verticalalignment='center')
            axes[1, 1].set_title('Performance Summary')
            axes[1, 1].axis('off')
            
            plt.tight_layout()
            
            # Save report
            report_path = os.path.join(self.config.output_folder, 'analysis_report.png')
            plt.savefig(report_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            return report_path
            
        except Exception as e:
            self.logger.error(f"Visualization generation error: {e}")
            return ""

def main():
    """Example usage of the enhanced analyzer"""
    config = AnalysisConfig(
        pose_confidence=0.7,
        hand_confidence=0.7,
        face_confidence=0.8,
        object_confidence=0.6,
        enable_visualization=True,
        generate_reports=True
    )
    
    analyzer = EnhancedAnalyzer(config)
    
    # Example video processing
    video_path = "sample_video.mp4"  # Replace with actual video path
    
    try:
        results = analyzer.process_video_enhanced(video_path)
        
        # Generate visualization report
        report_path = analyzer.generate_visualization_report(results)
        
        print(f"Analysis completed!")
        print(f"Total events detected: {results['statistics']['total_events']}")
        print(f"Processing time: {results['statistics']['processing_time']:.2f} seconds")
        print(f"Report saved to: {report_path}")
        
    except Exception as e:
        print(f"Analysis failed: {e}")

if __name__ == "__main__":
    main()