import os
import cv2
import pandas as pd
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
from mediapipe import solutions as mp_solutions
from geopy.distance import geodesic
import json

@dataclass
class DetectionConfig:
    """Configuration parameters for detection"""
    pose_confidence: float = 0.5
    hand_confidence: float = 0.5
    hand_tracking_confidence: float = 0.5
    smoothing_frames: int = 2
    cooldown_seconds: float = 2.0
    max_num_people: int = 10
    save_frames: bool = True
    output_folder: str = "detected_frames"
    # GPS validation parameters
    gps_tolerance_meters: float = 50.0  # Tolerance for GPS matching
    excel_file_path: str = "Detected_Signals_Lat_Long.xlsx"

@dataclass
class SignalLocation:
    """Represents a valid signal location"""
    latitude: float
    longitude: float
    index: int

@dataclass
class HandSignalDetection:
    """Represents a hand signal detection with validation"""
    timestamp: float
    video_time: str
    signal_type: str
    current_gps: Optional[Tuple[float, float]]
    nearest_valid_location: Optional[SignalLocation]
    distance_to_valid: Optional[float]
    is_violation: bool
    violation_reason: str
    frame_path: Optional[str]

class EnhancedHandSignalDetector:
    def __init__(self, config: DetectionConfig):
        """Initialize detector with GPS validation"""
        self.config = config
        
        # Initialize MediaPipe components
        self.holistic = mp_solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.pose_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
            enable_segmentation=False,
        )
        
        self.pose = mp_solutions.pose.Pose(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=config.pose_confidence,
            smooth_landmarks=True,
            enable_segmentation=False,
        )
        
        self.hands = mp_solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2 * config.max_num_people,
            min_detection_confidence=config.hand_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
        )

        # Detection buffers
        self.raised_hand_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]
        self.last_detection_times = [-float('inf')] * config.max_num_people

        # Load valid signal locations from Excel
        self.valid_locations = self.load_signal_locations()
        
        # Create output folder
        if config.save_frames and not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)

    def load_signal_locations(self) -> List[SignalLocation]:
        """Load valid signal locations from Excel file"""
        try:
            df = pd.read_excel(self.config.excel_file_path)
            locations = []
            for idx, row in df.iterrows():
                locations.append(SignalLocation(
                    latitude=row['LATITUDE'],
                    longitude=row['LONGITUDE'],
                    index=idx
                ))
            print(f"Loaded {len(locations)} valid signal locations from Excel")
            return locations
        except Exception as e:
            print(f"Error loading Excel file: {e}")
            return []

    def find_nearest_valid_location(self, current_gps: Tuple[float, float]) -> Tuple[Optional[SignalLocation], Optional[float]]:
        """Find the nearest valid signal location"""
        if not self.valid_locations or not current_gps:
            return None, None
        
        min_distance = float('inf')
        nearest_location = None
        
        for location in self.valid_locations:
            distance = geodesic(current_gps, (location.latitude, location.longitude)).meters
            if distance < min_distance:
                min_distance = distance
                nearest_location = location
        
        return nearest_location, min_distance

    def validate_signal_location(self, current_gps: Optional[Tuple[float, float]]) -> Tuple[bool, str, Optional[SignalLocation], Optional[float]]:
        """Validate if hand signal is raised at correct location"""
        if not current_gps:
            return True, "No GPS data available - cannot validate", None, None
        
        if not self.valid_locations:
            return True, "No valid locations loaded - cannot validate", None, None
        
        nearest_location, distance = self.find_nearest_valid_location(current_gps)
        
        if distance is None:
            return True, "Could not calculate distance", None, None
        
        is_valid = distance <= self.config.gps_tolerance_meters
        
        if is_valid:
            reason = f"Valid signal at authorized location (within {distance:.1f}m)"
        else:
            reason = f"VIOLATION: Signal raised {distance:.1f}m from nearest authorized location (tolerance: {self.config.gps_tolerance_meters}m)"
        
        return is_valid, reason, nearest_location, distance

    def is_hand_raised_and_extended(self, hand_landmarks, pose_landmarks) -> bool:
        """Check if hand is raised and extended"""
        if not hand_landmarks or not pose_landmarks:
            return False

        wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
        thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
        index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
        pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP]

        try:
            left_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER]
        except:
            return False

        # Hand height check
        hand_is_high = wrist.y < min(left_shoulder.y, right_shoulder.y)

        # Hand extension check
        thumb_vector = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
        index_vector = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
        pinky_vector = np.array([pinky_tip.x - wrist.x, pinky_tip.y - wrist.y])

        thumb_length = np.linalg.norm(thumb_vector)
        index_length = np.linalg.norm(index_vector)
        pinky_length = np.linalg.norm(pinky_vector)

        hand_is_extended = (thumb_length > 0.1 and index_length > 0.1 and pinky_length > 0.1)

        # Vertical orientation check
        hand_main_vector = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
        vertical_vector = np.array([0, -1])
        
        hand_main_vector = hand_main_vector / np.linalg.norm(hand_main_vector)
        angle = np.arccos(np.clip(np.dot(hand_main_vector, vertical_vector), -1.0, 1.0))
        hand_is_vertical = abs(angle) < np.pi/4

        return hand_is_high and hand_is_extended and hand_is_vertical

    def format_time(self, total_seconds: float) -> str:
        """Format time as HH:MM:SS"""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def get_current_gps_location(self, video_time_seconds: float) -> Optional[Tuple[float, float]]:
        """
        Get current GPS location based on video timestamp.
        This is a placeholder - implement based on your GPS data source.
        """
        # Placeholder implementation - replace with actual GPS data retrieval
        # For demo purposes, using a sample location
        if hasattr(self, 'sample_gps_data'):
            return self.sample_gps_data
        return None

    def set_sample_gps_data(self, latitude: float, longitude: float):
        """Set sample GPS data for testing"""
        self.sample_gps_data = (latitude, longitude)

    def process_video_with_gps_validation(self, input_path: str, gps_data: Optional[Dict] = None) -> List[HandSignalDetection]:
        """Process video with GPS validation for hand signals"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        detections = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                video_time_seconds = frame_count / fps

                # Convert to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

                # Process with MediaPipe
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)

                if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        if self.is_hand_raised_and_extended(hand_landmarks, pose_results.pose_landmarks):
                            person_id = hand_idx % self.config.max_num_people

                            # Update buffer
                            self.raised_hand_buffers[person_id].pop(0)
                            self.raised_hand_buffers[person_id].append(True)

                            # Check if consistently raised
                            if all(self.raised_hand_buffers[person_id]):
                                # Check cooldown
                                if (video_time_seconds - self.last_detection_times[person_id] >= self.config.cooldown_seconds):
                                    
                                    # Get current GPS location
                                    current_gps = self.get_current_gps_location(video_time_seconds)
                                    
                                    # Validate signal location
                                    is_valid, reason, nearest_location, distance = self.validate_signal_location(current_gps)
                                    
                                    # Save frame if enabled
                                    frame_path = None
                                    if self.config.save_frames:
                                        frame_filename = f"hand_signal_{self.format_time(video_time_seconds).replace(':', '-')}.jpg"
                                        frame_path = os.path.join(self.config.output_folder, frame_filename)
                                        cv2.imwrite(frame_path, frame)

                                    # Create detection record
                                    detection = HandSignalDetection(
                                        timestamp=video_time_seconds,
                                        video_time=self.format_time(video_time_seconds),
                                        signal_type="Hand Raised and Extended",
                                        current_gps=current_gps,
                                        nearest_valid_location=nearest_location,
                                        distance_to_valid=distance,
                                        is_violation=not is_valid,
                                        violation_reason=reason,
                                        frame_path=frame_path
                                    )
                                    
                                    detections.append(detection)
                                    self.last_detection_times[person_id] = video_time_seconds

                                    print(f"[{detection.video_time}] {detection.signal_type}")
                                    print(f"  GPS: {current_gps}")
                                    print(f"  Status: {reason}")
                                    if detection.is_violation:
                                        print(f"  ⚠️  VIOLATION DETECTED!")
                                    print()

                        else:
                            # Reset buffer if hand not raised
                            person_id = hand_idx % self.config.max_num_people
                            self.raised_hand_buffers[person_id] = [False] * self.config.smoothing_frames

        finally:
            cap.release()

        return detections

    def generate_violation_report(self, detections: List[HandSignalDetection]) -> Dict:
        """Generate a comprehensive violation report"""
        total_signals = len(detections)
        violations = [d for d in detections if d.is_violation]
        valid_signals = [d for d in detections if not d.is_violation]
        
        report = {
            "summary": {
                "total_hand_signals": total_signals,
                "valid_signals": len(valid_signals),
                "violations": len(violations),
                "violation_rate": len(violations) / total_signals * 100 if total_signals > 0 else 0
            },
            "violations": [
                {
                    "timestamp": v.video_time,
                    "gps_location": v.current_gps,
                    "nearest_valid_location": (v.nearest_valid_location.latitude, v.nearest_valid_location.longitude) if v.nearest_valid_location else None,
                    "distance_from_valid": v.distance_to_valid,
                    "reason": v.violation_reason,
                    "frame_saved": v.frame_path
                }
                for v in violations
            ],
            "valid_signals": [
                {
                    "timestamp": v.video_time,
                    "gps_location": v.current_gps,
                    "distance_from_valid": v.distance_to_valid
                }
                for v in valid_signals
            ]
        }
        
        return report

# Example usage
if __name__ == "__main__":
    # Configuration
    config = DetectionConfig(
        gps_tolerance_meters=50.0,
        excel_file_path="Detected_Signals_Lat_Long.xlsx"
    )
    
    # Initialize detector
    detector = EnhancedHandSignalDetector(config)
    
    # Set sample GPS data for testing (replace with actual GPS feed)
    detector.set_sample_gps_data(15.267390, 73.980355)  # Sample from Excel data
    
    # Process video
    video_path = "path/to/your/video.mp4"
    # detections = detector.process_video_with_gps_validation(video_path)
    
    # Generate report
    # report = detector.generate_violation_report(detections)
    # print(json.dumps(report, indent=2))