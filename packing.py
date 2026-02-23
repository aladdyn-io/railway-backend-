import cv2
import mediapipe as mp
from ultralytics import YOLO
import numpy as np
import csv
import os

# ---------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------
VIDEO_PATH = "C:/Users/Asus/Downloads/CH1_00000300099000000.mp4"
OUTPUT_DIR = "detected_actions"
CSV_PATH = "actions_log.csv"
OUTPUT_VIDEO = "output.mp4"

FIRST_MINUTES = 10
LAST_MINUTES = 10
CONF_THRESH = 0.45

# Bag class IDs/names in COCO: backpack, handbag, suitcase
BAG_CLASSES = ["backpack", "handbag", "suitcase"]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---------------------------------------------------------------
# Load Models
# ---------------------------------------------------------------
yolo = YOLO("yolov8s.pt")  # make sure yolov8s.pt is downloaded

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles

hands_model = mp_hands.Hands(
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------
def get_hand_center(hand_landmarks, frame_w, frame_h):
    xs, ys = [], []
    for lm in hand_landmarks.landmark:
        xs.append(int(lm.x * frame_w))
        ys.append(int(lm.y * frame_h))
    return int(np.mean(xs)), int(np.mean(ys))


def hand_near_bag(hand_xy, bag_boxes, threshold=80):
    """Check if hand is close to any bag box"""
    hx, hy = hand_xy
    for (x1, y1, x2, y2) in bag_boxes:
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        dist = np.sqrt((hx - cx) ** 2 + (hy - cy) ** 2)
        if dist <= threshold:
            return True
    return False


def detect_action(prev_pos, curr_pos, prev_in, curr_in):
    """
    Rule-based action detection:
    - moving into bag → hand moves towards bag
    - packing → hand moves into bag and disappears
    - unpacking → hand moves out of bag
    """
    if prev_pos is None or curr_pos is None:
        return "normal"

    prev_x, prev_y = prev_pos
    curr_x, curr_y = curr_pos

    # vertical motion
    movement = curr_y - prev_y

    # moving into bag
    if movement < 0 and not prev_in and curr_in:
        return "moving into bag"

    # packing
    if prev_in and not curr_in:
        return "packing"

    # unpacking
    if not prev_in and curr_in:
        return "unpacking"

    return "normal"


# ---------------------------------------------------------------
# CSV Setup
# ---------------------------------------------------------------
csv_file = open(CSV_PATH, 'w', newline='')
csv_writer = csv.writer(csv_file)
csv_writer.writerow(["timestamp", "frame_id", "action"])

# ---------------------------------------------------------------
# Video Setup
# ---------------------------------------------------------------
cap = cv2.VideoCapture(VIDEO_PATH)
fps = cap.get(cv2.CAP_PROP_FPS)
total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
duration_sec = total_frames / fps

first_frames = int(FIRST_MINUTES * 60 * fps)
last_frames = int(LAST_MINUTES * 60 * fps)
start_last = max(0, total_frames - last_frames)

print("Total Duration:", duration_sec)

# Video writer
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
out_video = cv2.VideoWriter(OUTPUT_VIDEO, fourcc, fps, (frame_w, frame_h))

# ---------------------------------------------------------------
# Frame Processing Function
# ---------------------------------------------------------------
def process_frames(start_frame, end_frame):
    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    frame_id = start_frame

    prev_hand_positions = []

    print(f"\n---- PROCESSING FRAMES {start_frame} TO {end_frame} ----\n")

    while frame_id < end_frame:
        ret, frame = cap.read()
        if not ret:
            break

        timestamp = cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0

        # -------------------------------
        # 1. YOLO BAG DETECTION
        # -------------------------------
        yolo_results = yolo(frame)[0]
        bag_boxes = []
        for box in yolo_results.boxes:
            cls_name = yolo_results.names[int(box.cls[0])]
            conf = float(box.conf[0])
            if conf < CONF_THRESH:
                continue
            if cls_name in BAG_CLASSES:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                bag_boxes.append((x1, y1, x2, y2))

        if len(bag_boxes) == 0:
            frame_id += 1
            out_video.write(frame)
            continue

        # -------------------------------
        # 2. HAND DETECTION (MediaPipe)
        # -------------------------------
        mp_results = hands_model.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        curr_hand_positions = []

        if mp_results.multi_hand_landmarks:
            for hand_landmarks in mp_results.multi_hand_landmarks:
                cx, cy = get_hand_center(hand_landmarks, frame_w, frame_h)
                curr_hand_positions.append((cx, cy))

                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS,
                    mp_styles.get_default_hand_landmarks_style(),
                    mp_styles.get_default_hand_connections_style()
                )

        # Reset logic if hand count changes
        if len(prev_hand_positions) != len(curr_hand_positions):
            prev_hand_positions = curr_hand_positions.copy()
            frame_id += 1
            out_video.write(frame)
            continue

        # -------------------------------
        # 3. ACTION DETECTION
        # -------------------------------
        actions = []
        for i in range(len(curr_hand_positions)):
            prev_pos = prev_hand_positions[i] if i < len(prev_hand_positions) else None
            curr_pos = curr_hand_positions[i]
            prev_in = hand_near_bag(prev_pos, bag_boxes) if prev_pos else False
            curr_in = hand_near_bag(curr_pos, bag_boxes)

            action = detect_action(prev_pos, curr_pos, prev_in, curr_in)
            actions.append(action)

        prev_hand_positions = curr_hand_positions.copy()

        # -------------------------------
        # 4. DRAW BAGS
        # -------------------------------
        for (x1, y1, x2, y2) in bag_boxes:
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 255), 2)
            cv2.putText(frame, "Bag", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Draw hand actions
        for i, (hx, hy) in enumerate(curr_hand_positions):
            cv2.circle(frame, (hx, hy), 8, (0, 0, 255), -1)
            cv2.putText(frame, actions[i], (hx + 10, hy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        # -------------------------------
        # 5. Write CSV
        # -------------------------------
        for i, action in enumerate(actions):
            csv_writer.writerow([timestamp, frame_id, action])

        # -------------------------------
        # 6. Write Frame to Video
        # -------------------------------
        out_video.write(frame)
        frame_id += 1


# ---------------------------------------------------------------
# RUN
# ---------------------------------------------------------------
print("\n---- FIRST 10 MINUTES ----")
process_frames(0, first_frames)

print("\n---- LAST 10 MINUTES ----")
process_frames(start_last, total_frames)

csv_file.close()
cap.release()
out_video.release()
print("\nProcessing Complete! Output video saved as:", OUTPUT_VIDEO)
