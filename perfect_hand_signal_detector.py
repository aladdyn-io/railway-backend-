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
class HandSignalRule:
    """Represents a hand signal rule from Excel"""
    location_id: int
    latitude: float
    longitude: float
    station_name: str
    signal_required: bool
    signal_type: str
    tolerance_meters: float
    mandatory: bool
    expected_time: Optional[float] = None  # Expected time in video (seconds)
    tolerance_seconds: float = 10.0  # ±10 seconds tolerance for timing

@dataclass
class HandSignalDetection:
    """Represents a detected hand signal with validation"""
    timestamp: float
    video_time: str
    signal_detected: bool
    signal_type: str
    current_gps: Optional[Tuple[float, float]]
    nearest_rule: Optional[HandSignalRule]
    distance_to_rule: Optional[float]
    compliance_status: str
    violation_type: str
    confidence: float

class PerfectHandSignalDetector:
    def __init__(self, excel_file_path: str = "Detected_Signals_Lat_Long_Enhanced.xlsx"):
        """Initialize with Excel rules"""
        self.excel_file_path = excel_file_path
        self.signal_rules = self.load_signal_rules()
        self.current_gps = None
        
        # MediaPipe configuration - stricter thresholds to reduce false positives
        self.holistic = mp_solutions.holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
            enable_segmentation=False,
        )
        
        self.hands = mp_solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=0.6,
            min_tracking_confidence=0.6,
        )
        
        # Detection buffers for stability - More lenient
        self.signal_buffer = []
        self.buffer_size = 3  # Require only 3 consecutive detections (was 5)
        self.last_detection_time = 0
        self.cooldown_period = 2.0  # 2 seconds between detections (was 3)
        
        print(f"🚂 Perfect Hand Signal Detector initialized with {len(self.signal_rules)} rules")

    def load_signal_rules(self) -> List[HandSignalRule]:
        """Load hand signal rules from Excel file"""
        try:
            df = pd.read_excel(self.excel_file_path)
            rules = []
            
            for idx, row in df.iterrows():
                # Use station name from Excel if available, otherwise generate
                station_name = row.get('STATION_NAME', f"Station_{idx+1:03d}")
                
                # Get tolerance from Excel if available
                tolerance_meters = row.get('TOLERANCE_METERS', 30.0)
                
                # Get signal required from Excel if available
                signal_required = row.get('SIGNAL_REQUIRED', True)
                
                # Get expected time and timing tolerance from Excel if available
                expected_time = row.get('EXPECTED_TIME_SECONDS', None)
                tolerance_seconds = row.get('TOLERANCE_SECONDS', 10.0)
                
                rule = HandSignalRule(
                    location_id=row.get('STATION_ID', idx),
                    latitude=row['LATITUDE'],
                    longitude=row['LONGITUDE'],
                    station_name=station_name,
                    signal_required=signal_required,
                    signal_type="raised_hand",
                    tolerance_meters=tolerance_meters,
                    mandatory=signal_required,
                    expected_time=expected_time,
                    tolerance_seconds=tolerance_seconds
                )
                rules.append(rule)
            
            print(f"✅ Loaded {len(rules)} hand signal rules from Excel")
            if 'STATION_NAME' in df.columns:
                print(f"   📍 Using station names from Excel")
            if 'EXPECTED_TIME_SECONDS' in df.columns:
                print(f"   ⏰ Timing rules available in Excel")
            return rules
            
        except Exception as e:
            print(f"❌ Error loading Excel rules: {e}")
            import traceback
            traceback.print_exc()
            return []

    def set_current_gps(self, latitude: float, longitude: float):
        """Set current GPS location"""
        self.current_gps = (latitude, longitude)
        print(f"📍 GPS set to: {latitude:.6f}, {longitude:.6f}")

    def find_applicable_rule(self, gps_location: Tuple[float, float]) -> Tuple[Optional[HandSignalRule], float]:
        """Find the nearest applicable signal rule"""
        if not self.signal_rules or not gps_location:
            return None, float('inf')
        
        min_distance = float('inf')
        nearest_rule = None
        
        for rule in self.signal_rules:
            distance = geodesic(gps_location, (rule.latitude, rule.longitude)).meters
            if distance < min_distance:
                min_distance = distance
                nearest_rule = rule
        
        return nearest_rule, min_distance

    def is_perfect_hand_signal(self, hand_landmarks, pose_landmarks) -> Tuple[bool, float]:
        """Enhanced hand signal detection with confidence scoring"""
        if not hand_landmarks or not pose_landmarks:
            return False, 0.0
        
        confidence_score = 0.0
        
        try:
            # Get landmarks
            wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
            thumb_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.THUMB_TIP]
            index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
            middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.PINKY_TIP]
            
            # Get pose landmarks
            left_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.LEFT_SHOULDER]
            right_shoulder = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.RIGHT_SHOULDER]
            nose = pose_landmarks.landmark[mp_solutions.pose.PoseLandmark.NOSE]
            
            # 1. Height Check - Hand must be above shoulders (more lenient)
            shoulder_avg_y = (left_shoulder.y + right_shoulder.y) / 2
            
            # Stricter: hand must be noticeably above shoulder level (wrist.y < shoulder - 0.03)
            if wrist.y < shoulder_avg_y - 0.03:
                confidence_score += 30.0
                if wrist.y < shoulder_avg_y - 0.15:  # 15% above shoulders
                    confidence_score += 20.0
                elif wrist.y < shoulder_avg_y - 0.1:  # 10% above shoulders
                    confidence_score += 10.0
            
            # 2. Hand Extension Check - More lenient finger extension
            finger_tips = [thumb_tip, index_tip, middle_tip, ring_tip, pinky_tip]
            extended_fingers = 0
            
            for tip in finger_tips:
                distance_from_wrist = np.sqrt((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)
                if distance_from_wrist > 0.10:  # Stricter: require clearer finger extension
                    extended_fingers += 1
            
            extension_score = (extended_fingers / 5.0) * 35.0  # Increased weight
            confidence_score += extension_score
            
            # 3. Vertical Orientation Check
            hand_vector = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
            vertical_vector = np.array([0, -1])
            
            if np.linalg.norm(hand_vector) > 0:
                hand_vector = hand_vector / np.linalg.norm(hand_vector)
                angle = np.arccos(np.clip(np.dot(hand_vector, vertical_vector), -1.0, 1.0))
                
                # Perfect vertical gets full points
                if angle < np.pi/6:  # Within 30 degrees
                    confidence_score += 20.0
                elif angle < np.pi/4:  # Within 45 degrees
                    confidence_score += 10.0
            
            # 4. Stability Check - Hand should be relatively stable
            hand_center_x = (thumb_tip.x + index_tip.x + middle_tip.x + ring_tip.x + pinky_tip.x) / 5
            hand_center_y = (thumb_tip.y + index_tip.y + middle_tip.y + ring_tip.y + pinky_tip.y) / 5
            
            # Check if hand is centered and stable
            if 0.3 < hand_center_x < 0.7 and hand_center_y < shoulder_avg_y:
                confidence_score += 10.0
            
            # 5. Clear visibility bonus
            if all(tip.visibility > 0.8 for tip in finger_tips if hasattr(tip, 'visibility')):
                confidence_score += 10.0
            
            # Normalize confidence to 0-100
            confidence_score = min(100.0, confidence_score)
            
            # Stricter threshold: Signal is valid if confidence >= 65%
            # Reduces false positives when pilot is just sitting
            is_valid = confidence_score >= 65.0
            
            return is_valid, confidence_score
            
        except Exception as e:
            print(f"❌ Hand signal analysis error: {e}")
            return False, 0.0

    def validate_signal_compliance(self, gps_location: Optional[Tuple[float, float]], 
                                 signal_detected: bool, confidence: float, 
                                 timestamp_seconds: float = 0) -> HandSignalDetection:
        """Comprehensive signal compliance validation - Checks BOTH location AND timing"""
        
        if not gps_location:
            return HandSignalDetection(
                timestamp=timestamp_seconds,
                video_time=self.format_time(timestamp_seconds),
                signal_detected=signal_detected,
                signal_type="unknown",
                current_gps=None,
                nearest_rule=None,
                distance_to_rule=None,
                compliance_status="NO_GPS",
                violation_type="No GPS data available",
                confidence=confidence
            )
        
        # Find applicable rule
        nearest_rule, distance = self.find_applicable_rule(gps_location)
        
        if not nearest_rule:
            return HandSignalDetection(
                timestamp=timestamp_seconds,
                video_time=self.format_time(timestamp_seconds),
                signal_detected=signal_detected,
                signal_type="unknown",
                current_gps=gps_location,
                nearest_rule=None,
                distance_to_rule=None,
                compliance_status="NO_RULES",
                violation_type="No signal rules found",
                confidence=confidence
            )
        
        # Check location compliance
        within_location_tolerance = distance <= nearest_rule.tolerance_meters
        
        # Check timing compliance if expected time is available
        timing_compliant = True
        time_violation = None
        expected_time = nearest_rule.expected_time
        
        # Check timing if expected time is set
        if expected_time is not None and expected_time > 0:
            time_diff = abs(timestamp_seconds - expected_time)
            tolerance_seconds = nearest_rule.tolerance_seconds
            timing_compliant = time_diff <= tolerance_seconds
            
            if not timing_compliant:
                if timestamp_seconds > expected_time:
                    time_violation = f"LATE: {time_diff:.1f}s after expected time (tolerance: ±{tolerance_seconds}s)"
                else:
                    time_violation = f"EARLY: {time_diff:.1f}s before expected time (tolerance: ±{tolerance_seconds}s)"
        
        # Determine compliance status - BOTH location AND timing must be correct
        if not nearest_rule.signal_required:
            # No signal required at this location
            compliance_status = "COMPLIANT"
            violation_type = "None - No signal required at this location"
        elif within_location_tolerance and signal_detected and timing_compliant and confidence >= 50.0:
            # ✅ PERFECT: Correct location, signal detected, correct timing, good confidence
            compliance_status = "COMPLIANT"
            violation_type = "None"
        elif within_location_tolerance and not signal_detected:
            # ❌ MISSED: At correct location but no signal detected
            compliance_status = "VIOLATION"
            violation_type = f"MISSED SIGNAL - Required at {nearest_rule.station_name} but not detected"
        elif within_location_tolerance and signal_detected and not timing_compliant:
            # ❌ TIMING ERROR: Signal detected at correct location but wrong time
            compliance_status = "VIOLATION"
            violation_type = f"TIMING ERROR - {time_violation} at {nearest_rule.station_name}"
        elif within_location_tolerance and signal_detected and confidence < 50.0:
            # ❌ POOR QUALITY: Signal detected but low confidence
            compliance_status = "VIOLATION" 
            violation_type = f"POOR SIGNAL QUALITY - Confidence {confidence:.1f}% < 50%"
        elif not within_location_tolerance and signal_detected:
            # ❌ WRONG LOCATION: Signal detected but not at correct station
            compliance_status = "VIOLATION"
            violation_type = f"WRONG LOCATION - {distance:.1f}m from {nearest_rule.station_name} (tolerance: {nearest_rule.tolerance_meters}m)"
        else:
            # No signal and not required - OK
            compliance_status = "COMPLIANT"
            violation_type = "None - No signal required at this location"
        
        return HandSignalDetection(
            timestamp=timestamp_seconds,
            video_time=self.format_time(timestamp_seconds),
            signal_detected=signal_detected,
            signal_type="raised_hand" if signal_detected else "none",
            current_gps=gps_location,
            nearest_rule=nearest_rule,
            distance_to_rule=distance,
            compliance_status=compliance_status,
            violation_type=violation_type,
            confidence=confidence
        )

    def process_frame(self, frame, timestamp_seconds: float) -> Optional[HandSignalDetection]:
        """Process single frame for hand signal detection"""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Process with MediaPipe
        holistic_results = self.holistic.process(rgb_frame)
        hand_results = self.hands.process(rgb_frame)
        
        # Debug: Log what MediaPipe detected
        has_pose = holistic_results.pose_landmarks is not None
        has_hands = hand_results.multi_hand_landmarks is not None and len(hand_results.multi_hand_landmarks) > 0
        num_hands = len(hand_results.multi_hand_landmarks) if hand_results.multi_hand_landmarks else 0
        
        if timestamp_seconds % 5 < 0.1:  # Log every 5 seconds
            print(f"🔍 Frame at {timestamp_seconds:.1f}s: Pose={has_pose}, Hands={num_hands}")
        
        signal_detected = False
        max_confidence = 0.0
        
        # Try to detect with pose landmarks first (more accurate)
        if holistic_results.pose_landmarks and hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                is_signal, confidence = self.is_perfect_hand_signal(
                    hand_landmarks, holistic_results.pose_landmarks
                )
                
                if is_signal and confidence > max_confidence:
                    signal_detected = True
                    max_confidence = confidence
        
        # Fallback: If no pose detected, try stricter hand-only detection
        elif hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                wrist = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.WRIST]
                # Stricter: wrist in upper third of frame (hand clearly raised)
                if wrist.y < 0.4:
                    index_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.INDEX_FINGER_TIP]
                    middle_tip = hand_landmarks.landmark[mp_solutions.hands.HandLandmark.MIDDLE_FINGER_TIP]
                    distance = np.sqrt((index_tip.x - wrist.x)**2 + (index_tip.y - wrist.y)**2)
                    if distance > 0.10:  # Stricter: clearer finger extension
                        signal_detected = True
                        max_confidence = 60.0
                        print(f"🤚 Hand signal detected (hand-only mode) at {timestamp_seconds:.1f}s")
                        break
        
        # Update signal buffer for stability
        self.signal_buffer.append(signal_detected)
        if len(self.signal_buffer) > self.buffer_size:
            self.signal_buffer.pop(0)
        
        # Stricter: Require 3 out of 3 consecutive detections for stability
        stable_signal = len(self.signal_buffer) >= self.buffer_size and sum(self.signal_buffer) >= 3
        
        # Allow immediate detection only if confidence is high (>= 75%)
        high_confidence_detection = signal_detected and max_confidence >= 75.0
        can_detect = stable_signal or high_confidence_detection
        
        # Debug logging for detection attempts
        if signal_detected and timestamp_seconds % 2 < 0.1:  # Log every 2 seconds when signal detected
            print(f"🤚 Signal detected at {timestamp_seconds:.1f}s: confidence={max_confidence:.1f}%, buffer={sum(self.signal_buffer)}/{len(self.signal_buffer)}, stable={stable_signal}")
        
        if can_detect and (timestamp_seconds - self.last_detection_time) >= self.cooldown_period:
            self.last_detection_time = timestamp_seconds
            
            # Validate compliance - Pass timestamp for timing validation
            detection = self.validate_signal_compliance(self.current_gps, True, max_confidence, timestamp_seconds)
            
            # Log detection
            status_icon = "✅" if detection.compliance_status == "COMPLIANT" else "❌"
            print(f"\n{status_icon} [{detection.video_time}] HAND SIGNAL DETECTED - {detection.compliance_status}")
            print(f"   📍 GPS: {self.current_gps}")
            print(f"   🎯 Confidence: {detection.confidence:.1f}%")
            print(f"   📏 Distance: {detection.distance_to_rule:.1f}m" if detection.distance_to_rule else "   📏 Distance: N/A")
            if detection.compliance_status == "VIOLATION":
                print(f"   ⚠️  VIOLATION: {detection.violation_type}")
            print()
            
            return detection
        
        return None

    def format_time(self, total_seconds: float) -> str:
        """Format time as HH:MM:SS"""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

    def generate_compliance_report(self, detections: List[HandSignalDetection]) -> Dict:
        """Generate comprehensive compliance report"""
        total_signals = len(detections)
        compliant_signals = [d for d in detections if d.compliance_status == "COMPLIANT"]
        violations = [d for d in detections if d.compliance_status == "VIOLATION"]
        
        violation_types = {}
        for v in violations:
            violation_types[v.violation_type] = violation_types.get(v.violation_type, 0) + 1
        
        report = {
            "summary": {
                "total_hand_signals": total_signals,
                "compliant_signals": len(compliant_signals),
                "violations": len(violations),
                "compliance_rate": (len(compliant_signals) / total_signals * 100) if total_signals > 0 else 100.0,
                "average_confidence": sum(d.confidence for d in detections) / total_signals if total_signals > 0 else 0.0
            },
            "violations_by_type": violation_types,
            "detailed_violations": [
                {
                    "timestamp": v.video_time,
                    "type": v.violation_type,
                    "location": v.current_gps,
                    "nearest_station": v.nearest_rule.station_name if v.nearest_rule else "Unknown",
                    "distance": v.distance_to_rule,
                    "confidence": v.confidence
                }
                for v in violations
            ],
            "excel_rules_applied": len(self.signal_rules),
            "gps_validation_active": self.current_gps is not None
        }
        
        return report

# Test the perfect hand signal detector
if __name__ == "__main__":
    detector = PerfectHandSignalDetector()
    
    # Set test GPS location
    detector.set_current_gps(15.267390, 73.980355)
    
    print("🎯 Perfect Hand Signal Detection System Ready!")
    print(f"📋 Rules loaded: {len(detector.signal_rules)}")
    print(f"📍 GPS location set: {detector.current_gps}")
    print("🚂 System optimized for railway safety compliance!")