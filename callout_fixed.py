import os
import cv2
import time
import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
from mediapipe import solutions as mp_solutions

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

class GestureDetector:
    def __init__(self, config: DetectionConfig):
        """Initialize detector with given configuration"""
        self.config = config
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
            smooth_segmentation=False,
        )
        self.hands = mp_solutions.hands.Hands(
            static_image_mode=False,
            max_num_hands=2 * config.max_num_people,
            min_detection_confidence=config.hand_confidence,
            min_tracking_confidence=config.hand_tracking_confidence,
        )

        self.raised_hand_buffers = [[False] * config.smoothing_frames for _ in range(config.max_num_people)]
        self.last_detection_times = [-float('inf')] * config.max_num_people

        if config.save_frames and not os.path.exists(config.output_folder):
            os.makedirs(config.output_folder)
            print(f"Created output folder: {config.output_folder}")

    def format_time(self, total_seconds: float) -> str:
        """Format time with hours, minutes, and seconds."""
        hours = int(total_seconds // 3600)
        minutes = int((total_seconds % 3600) // 60)
        seconds = int(total_seconds % 60)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d}"

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

        hand_is_high = wrist.y < min(left_shoulder.y, right_shoulder.y)

        thumb_vector = np.array([thumb_tip.x - wrist.x, thumb_tip.y - wrist.y])
        index_vector = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
        pinky_vector = np.array([pinky_tip.x - wrist.x, pinky_tip.y - wrist.y])

        thumb_length = np.linalg.norm(thumb_vector)
        index_length = np.linalg.norm(index_vector)
        pinky_length = np.linalg.norm(pinky_vector)

        hand_is_extended = (thumb_length > 0.1 and index_length > 0.1 and pinky_length > 0.1)

        hand_main_vector = np.array([index_tip.x - wrist.x, index_tip.y - wrist.y])
        vertical_vector = np.array([0, -1])

        if np.linalg.norm(hand_main_vector) == 0:
            return False

        hand_main_vector = hand_main_vector / np.linalg.norm(hand_main_vector)
        vertical_vector = vertical_vector / np.linalg.norm(vertical_vector)

        angle = np.arccos(np.clip(np.dot(hand_main_vector, vertical_vector), -1.0, 1.0))
        hand_is_vertical = abs(angle) < np.pi/4

        return hand_is_high and hand_is_extended and hand_is_vertical

    def draw_landmarks(self, frame, multi_pose_landmarks=None, multi_hand_landmarks=None):
        """Draw landmarks on frame"""
        annotated_frame = frame.copy()

        if multi_pose_landmarks:
            for pose_landmarks in multi_pose_landmarks:
                mp_solutions.drawing_utils.draw_landmarks(
                    annotated_frame,
                    pose_landmarks,
                    mp_solutions.holistic.POSE_CONNECTIONS,
                    mp_solutions.drawing_styles.get_default_pose_landmarks_style()
                )

        if multi_hand_landmarks:
            for hand_landmarks in multi_hand_landmarks:
                mp_solutions.drawing_utils.draw_landmarks(
                    annotated_frame,
                    hand_landmarks,
                    mp_solutions.holistic.HAND_CONNECTIONS,
                    mp_solutions.drawing_styles.get_default_hand_landmarks_style(),
                    mp_solutions.drawing_styles.get_default_hand_connections_style()
                )

        return annotated_frame

    def save_frame(self, frame, video_time_seconds, signal_type, person_id=None):
        """Save frame with timestamp"""
        if not self.config.save_frames:
            return None

        hours = int(video_time_seconds // 3600)
        minutes = int((video_time_seconds % 3600) // 60)
        seconds = int(video_time_seconds % 60)

        timestamp = f"{hours:02d}_{minutes:02d}_{seconds:02d}"
        signal_tag = signal_type.replace(" ", "_").lower()
        person_info = f"_person{person_id}" if person_id is not None else ""
        filename = f"{timestamp}_{signal_tag}{person_info}.jpg"

        filepath = os.path.join(self.config.output_folder, filename)
        cv2.imwrite(filepath, frame)
        return filepath

    def process_video(self, input_path: str) -> List[Tuple[float, str, Optional[str]]]:
        """Process video for hand detection"""
        if not os.path.exists(input_path):
            raise FileNotFoundError(f"Input video not found: {input_path}")

        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            raise RuntimeError("Failed to open video file")

        detection_log = []
        fps = cap.get(cv2.CAP_PROP_FPS)

        try:
            frame_count = 0

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                frame_count += 1
                video_time_seconds = frame_count / fps

                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pose_results = self.pose.process(rgb_frame)
                hand_results = self.hands.process(rgb_frame)

                if pose_results.pose_landmarks and hand_results.multi_hand_landmarks:
                    for hand_idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                        if self.is_hand_raised_and_extended(hand_landmarks, pose_results.pose_landmarks):
                            person_id = hand_idx % self.config.max_num_people

                            self.raised_hand_buffers[person_id].pop(0)
                            self.raised_hand_buffers[person_id].append(True)

                            if all(self.raised_hand_buffers[person_id]):
                                if (video_time_seconds - self.last_detection_times[person_id] >= self.config.cooldown_seconds):
                                    signal_type = "Hand Raised and Extended"
                                    
                                    annotated_frame = self.draw_landmarks(
                                        frame,
                                        [pose_results.pose_landmarks],
                                        [hand_landmarks]
                                    )

                                    frame_path = None
                                    if self.config.save_frames:
                                        frame_path = self.save_frame(
                                            annotated_frame, 
                                            video_time_seconds, 
                                            signal_type, 
                                            person_id
                                        )

                                    formatted_time = self.format_time(video_time_seconds)

                                    detection_log.append((
                                        video_time_seconds,
                                        f"Signal detected at {formatted_time}: {signal_type} (Person {person_id})",
                                        frame_path
                                    ))

                                    self.last_detection_times[person_id] = video_time_seconds

        finally:
            cap.release()

        return detection_log

def save_log_to_file(log: List[Tuple[float, str, Optional[str]]], output_path: str):
    """Save detection log to file"""
    with open(output_path, 'w') as file:
        sorted_log = sorted(log, key=lambda x: x[0])
        for timestamp, message, frame_path in sorted_log:
            if frame_path:
                file.write(f"{message} [Frame saved: {frame_path}]\n")
            else:
                file.write(f"{message}\n")

def main():
    """Main entry point"""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_folder = f"detected_frames_{timestamp}"

    config = DetectionConfig(
        pose_confidence=0.5,
        hand_confidence=0.5,
        hand_tracking_confidence=0.5,
        smoothing_frames=2,
        cooldown_seconds=1.0,
        max_num_people=10,
        save_frames=True,
        output_folder=output_folder
    )

    detector = GestureDetector(config)

    try:
        import sys
        video_path = sys.argv[1] if len(sys.argv) > 1 else "test_video.mp4"

        if not os.path.exists(video_path):
            print(f"Error: Video file not found at {video_path}")
            print("Usage: python callout.py <video_path>")
            return

        print(f"Processing video: {video_path}")
        detection_log = detector.process_video(video_path)

        if detection_log:
            log_file_path = os.path.join(output_folder, 'detection_log.txt')
            save_log_to_file(detection_log, log_file_path)

            print(f"Detected {len(detection_log)} signals in the video.")
            print("Detection Log saved to:", log_file_path)
            print(f"Detected frames saved to: {output_folder}")

            signal_counts = {}
            person_counts = {}

            for _, message, _ in detection_log:
                signal_type = message.split(": ")[1].split(" (")[0]
                person_id = message.split("(Person ")[1].split(")")[0]

                signal_counts[signal_type] = signal_counts.get(signal_type, 0) + 1
                person_counts[person_id] = person_counts.get(person_id, 0) + 1

            print("\nSignal Type Breakdown:")
            for signal_type, count in signal_counts.items():
                print(f"  {signal_type}: {count}")

            print("\nDetections Per Person:")
            for person_id, count in person_counts.items():
                print(f"  Person {person_id}: {count}")
        else:
            print("No hand signals detected in the video.")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()