import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import csv
import os
import time

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
VIDEO_PATH = "C:/Users/Asus/Downloads/CH1_00000300099000000.mp4"
CSV_PATH = "actions_log_final.csv" # Updated CSV path
CONF_THRESH = 0.45

# Process only the first 10 minutes and last 10 minutes
FIRST_MINUTES = 10
LAST_MINUTES = 10

# Bag class IDs/names in COCO: backpack (24), handbag (25), suitcase (26)
BAG_CLASSES_YOLO_IDS = [24, 25, 26]

# Determine the device for YOLO (GPU preferred)
YOLO_DEVICE = 0 if cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'

# ---------------------------------------------------------------
# Load Models (Only once)
# ---------------------------------------------------------------
# Load YOLO model
yolo = YOLO("yolov8s.pt") 

# MediaPipe setup (using the new, faster style)
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

# Note: static_image_mode=False (the default) is faster for video streams
hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------------------
# Helper functions (Optimized)
# ---------------------------------------------------------------
def get_hand_center(hand_landmarks, frame_w, frame_h):
    """Calculates the center of a hand from its landmarks."""
    # Use array comprehension for potentially faster calculation
    xs = [lm.x * frame_w for lm in hand_landmarks.landmark]
    ys = [lm.y * frame_h for lm in hand_landmarks.landmark]
    return int(np.mean(xs)), int(np.mean(ys))

def hand_near_bag(hand_xy, bag_boxes, threshold_sq=80**2):
    """Check if hand is close to any bag box using squared distance."""
    hx, hy = hand_xy
    for x1, y1, x2, y2 in bag_boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        # Use squared distance to avoid expensive square root
        dist_sq = (hx - cx) ** 2 + (hy - cy) ** 2
        if dist_sq <= threshold_sq:
            return True
    return False

def detect_action(prev_pos, curr_pos, prev_in, curr_in):
    """Rule-based action detection (Same logic)."""
    if prev_pos is None or curr_pos is None:
        return "normal"

    # packing: hand was near, now is not (disappears/leaves)
    if prev_in and not curr_in:
        return "packing"

    # unpacking: hand was NOT near, now IS near (appears/enters)
    if not prev_in and curr_in:
        return "unpacking"
        
    # moving into bag (upward movement when entering near-bag zone)
    if not prev_in and curr_in and curr_pos[1] < prev_pos[1]:
         return "moving into bag"
    
    return "normal"

# ---------------------------------------------------------------
# Main Execution using YOLO Stream for high performance
# ---------------------------------------------------------------
def run_fast_log_only():
    start_time = time.time()
    
    # ---------------------------------------------------------------
    # Video & Range Setup
    # ---------------------------------------------------------------
    cap = cv2.VideoCapture(VIDEO_PATH)
    if not cap.isOpened():
        print(f"Error: Could not open video {VIDEO_PATH}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release() # Release cap, YOLO will handle video loading

    first_frames = int(FIRST_MINUTES * 60 * fps)
    last_frames = int(LAST_MINUTES * 60 * fps)
    start_last = max(0, total_frames - last_frames)

    # Calculate frame ranges to process
    process_ranges = []
    # If the video is long enough, split the ranges
    if first_frames + last_frames < total_frames:
        process_ranges.append((0, first_frames))
        process_ranges.append((start_last, total_frames))
    else: # If video is short, just process the whole thing
        process_ranges.append((0, total_frames))

    print(f"YOLO Device: {'GPU' if YOLO_DEVICE != 'cpu' else 'CPU'}")
    print(f"Processing Ranges (Frame ID): {process_ranges}")
    
    # ---------------------------------------------------------------
    # Processing Loop
    # ---------------------------------------------------------------
    all_log_entries = []
    prev_hand_positions = []
    frame_id = 0 # Initialize a global frame counter

    # Use YOLO's built-in video processing with stream=True for max speed.
    # We pass the full video and then conditionally process frames based on frame_id.
    
    # model.predict() with stream=True returns a generator
    yolo_results_generator = yolo.predict(
        source=VIDEO_PATH, 
        conf=CONF_THRESH, 
        classes=BAG_CLASSES_YOLO_IDS,
        device=YOLO_DEVICE,
        stream=True,         # CRITICAL: Enables memory-efficient generator for fast video processing
        verbose=False
    )
    
    for yolo_result in yolo_results_generator:
        
        # 1. Determine if the current frame_id falls within our target ranges
        is_in_range = False
        for start_f, end_f in process_ranges:
            if start_f <= frame_id < end_f:
                is_in_range = True
                break
        
        if not is_in_range:
            frame_id += 1
            # Reset tracking state if we jump out of a range
            prev_hand_positions = [] 
            continue 

        # Get frame data and timestamp from YOLO result object
        frame = yolo_result.orig_img
        # YOLO doesn't directly give frame_id/timestamp like cv2.VideoCapture in this mode.
        # We estimate timestamp from frame_id, assuming constant FPS.
        timestamp = frame_id / fps 

        # -------------------------------
        # 2. Extract YOLO Boxes
        # -------------------------------
        bag_boxes = []
        for box in yolo_result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            bag_boxes.append((x1, y1, x2, y2))
            
        # -------------------------------
        # 3. HAND DETECTION (MediaPipe)
        # -------------------------------
        # MediaPipe still needs to run on the frame.
        mp_results = hands_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hand_positions = []

        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                cx, cy = get_hand_center(hand_landmarks, frame_w, frame_h)
                curr_hand_positions.append((cx, cy))

        # Reset logic if hand count changes
        if len(prev_hand_positions) != len(curr_hand_positions):
            prev_hand_positions = curr_hand_positions.copy()
            frame_id += 1
            continue
        
        # -------------------------------
        # 4. ACTION DETECTION & LOGGING
        # -------------------------------
        for i in range(len(curr_hand_positions)):
            prev_pos = prev_hand_positions[i] if i < len(prev_hand_positions) else None
            curr_pos = curr_hand_positions[i]
            prev_in = hand_near_bag(prev_pos, bag_boxes) if prev_pos else False
            curr_in = hand_near_bag(curr_pos, bag_boxes)

            action = detect_action(prev_pos, curr_pos, prev_in, curr_in)
            
            # Log CSV
            all_log_entries.append((timestamp, frame_id, action))

        prev_hand_positions = curr_hand_positions.copy()
        frame_id += 1


    # ---------------------------------------------------------------
    # CSV Writing (Done once at the end to minimize I/O overhead)
    # ---------------------------------------------------------------
    with open(CSV_PATH, 'w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(["timestamp", "frame_id", "action"])
        csv_writer.writerows(all_log_entries)

    end_time = time.time()
    
    print("\nProcessing Complete!")
    print(f"Total time taken: {end_time - start_time:.2f} seconds.")
    print("Action log saved as:", CSV_PATH)


# ---------------------------------------------------------------
# RUN
# ---------------------------------------------------------------
if __name__ == '__main__':
    run_fast_log_only()