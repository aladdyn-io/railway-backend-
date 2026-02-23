import torch
import torchvision.models as models
import cv2
import os
import pandas as pd
import numpy as np

# === Configuration ===
MODEL_PATH = 'nitymed_resnet18.pth'  # Path to your trained model weights
INPUT_VIDEO_PATH = "C:/Users/Asus/Downloads/CH4_00000301449000000.mp4"
OUTPUT_FRAMES_DIR = 'detected_sleep_frames'  # Folder to save detected frames (microsleep/yawning)
LOG_CSV_PATH = 'detection_log.csv'

NUM_CLASSES = 3  # alert, microsleep, yawning

# === Load model and weights ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = models.resnet18(weights=None)
model.fc = torch.nn.Linear(model.fc.in_features, NUM_CLASSES)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model = model.to(device)
model.eval()

# === Preprocessing function ===
def preprocess(frame):
    """
    Preprocess frame for ResNet18:
    - Resize to 224x224
    - Convert BGR to RGB
    - Normalize using ImageNet mean/std
    - Convert to torch tensor with batch dim
    """
    means = np.array([0.485, 0.456, 0.406])
    stds = np.array([0.229, 0.224, 0.225])

    frame = cv2.resize(frame, (224, 224))
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame = frame.astype(np.float32) / 255.0
    frame = (frame - means) / stds
    frame = np.transpose(frame, (2, 0, 1))  # HWC to CHW
    tensor = torch.from_numpy(frame).unsqueeze(0).to(device)  # Add batch dimension and move to device
    return tensor

# === Class index to label ===
CLASS_MAPPING = {0: 'alert', 1: 'microsleep', 2: 'yawning'}

# === Prepare output directory ===
os.makedirs(OUTPUT_FRAMES_DIR, exist_ok=True)

# === Open video capture ===
cap = cv2.VideoCapture(INPUT_VIDEO_PATH)
if not cap.isOpened():
    raise IOError(f"Cannot open video file: {INPUT_VIDEO_PATH}")

fps = cap.get(cv2.CAP_PROP_FPS)
frame_count = 0
log_records = []

print("Starting video processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    # Preprocess and predict
    input_tensor = preprocess(frame)
    with torch.no_grad():
        outputs = model(input_tensor)
        pred_class = outputs.argmax(dim=1).item()

    # Save frame and log only if microsleep or yawning detected
    if pred_class in [1, 2]:
        label = CLASS_MAPPING[pred_class]
        timestamp = frame_count / fps
        filename = f"frame_{frame_count:06d}_{label}.jpg"
        filepath = os.path.join(OUTPUT_FRAMES_DIR, filename)
        cv2.imwrite(filepath, frame)

        log_records.append({
            "frame_number": frame_count,
            "timestamp_sec": timestamp,
            "label": label,
            "frame_path": filepath
        })

    # Optional: print progress every 100 frames
    if frame_count % 100 == 0:
        print(f"Processed {frame_count} frames...")

cap.release()
print("Video processing completed.")

# === Save log to CSV file ===
df_log = pd.DataFrame(log_records)
df_log.to_csv(LOG_CSV_PATH, index=False)
print(f"Detection log saved to {LOG_CSV_PATH}")

# === Print summary ===
total_duration_sec = frame_count / fps
microsleep_count = df_log['label'].value_counts().get('microsleep', 0)
yawning_count = df_log['label'].value_counts().get('yawning', 0)
event_frames = len(df_log)

print("\n--- Summary ---")
print(f"Total video duration: {total_duration_sec:.2f} seconds")
print(f"Total frames processed: {frame_count}")
print(f"Microsleep frames detected: {microsleep_count}")
print(f"Yawning frames detected: {yawning_count}")
print(f"Total sleep/yawning event frames saved: {event_frames}")

