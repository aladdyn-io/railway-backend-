import cv2
import mediapipe as mp
import numpy as np
import os
import time


# --- Parameters (adjust for your environment/fps) ---
EYE_AR_THRESH = 0.27
EYE_AR_CONSEC_FRAMES = 10  # Number of consecutive frames an eye must be below the threshold
HEAD_NOD_ANGLE_THRESH = 15  # Threshold for pitch angle (degrees) to detect head drop
FACE_ABSENT_CONSEC_FRAMES = 20  # Number of consecutive frames with no face detection


# --- Paths and Setup ---
video_path = "C:/Users/Asus/Downloads/CH4_00000301449000000.mp4"
save_dir = 'microsleep_frames'
os.makedirs(save_dir, exist_ok=True)


# --- MediaPipe setup ---
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
mp_face_detection = mp.solutions.face_detection
face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.5)


LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]


MODEL_POINTS_3D = np.array([
    (0.0, 0.0, 0.0),  # Nose Tip (1)
    (0.0, -63.6, -12.5),  # Chin (152)
    (43.3, 32.7, -26.0),  # Left Eye Corner (263)
    (-43.3, 32.7, -26.0),  # Right Eye Corner (33)
    (28.9, -28.9, -24.1),  # Left Mouth Corner (287)
    (-28.9, -28.9, -24.1)  # Right Mouth Corner (57)
])
POSE_LANDMARKS_IDX = [1, 152, 263, 33, 287, 57]


class SimpleLandmark:
    def __init__(self, x, y):
        self.x = int(x)
        self.y = int(y)


def eye_aspect_ratio(landmarks, indices):
    p = [np.array([landmarks[i].x, landmarks[i].y]) for i in indices]
    A = np.linalg.norm(p[1] - p[5])
    B = np.linalg.norm(p[2] - p[4])
    C = np.linalg.norm(p[0] - p[3])
    ear = (A + B) / (2.0 * C)
    return ear


def get_head_pose(landmarks, img_shape):
    image_points = np.array([
        (landmarks[idx].x, landmarks[idx].y) for idx in POSE_LANDMARKS_IDX
    ], dtype="double")

    h, w = img_shape[:2]
    focal_length = w
    center = (w // 2, h // 2)

    camera_matrix = np.array([
        [focal_length, 0, center[0]],
        [0, focal_length, center[1]],
        [0, 0, 1]
    ], dtype="double")

    dist_coeffs = np.zeros((4, 1))

    success, rotation_vector, translation_vector = cv2.solvePnP(
        MODEL_POINTS_3D, image_points, camera_matrix,
        dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE
    )
    if not success:
        return None, None, None, None

    rotation_mat, _ = cv2.Rodrigues(rotation_vector)
    pose_mat = np.hstack((rotation_mat, translation_vector))
    _, _, _, _, _, _, euler_angles = cv2.decomposeProjectionMatrix(pose_mat)

    pitch, yaw, roll = [angle[0] for angle in euler_angles]

    # Adjust pitch sign if needed (pitch positive means head down — check empirically)
    pitch = -pitch  # Flip sign to match "head nod down" positive

    nose_end_point2D, _ = cv2.projectPoints(
        np.array([(0.0, 0.0, 1000.0)]),
        rotation_vector, translation_vector, camera_matrix, dist_coeffs
    )

    return pitch, yaw, roll, (landmarks[1].x, landmarks[1].y,
                              int(nose_end_point2D[0][0][0]),
                              int(nose_end_point2D[0][0][1]))


cap = cv2.VideoCapture(video_path)
frame_count = 0
microsleep_counter = 0
face_absent_counter = 0
saved_frames = 0

print("Starting video processing...")

while True:
    ret, frame = cap.read()
    if not ret:
        print("\nEnd of video stream.")
        break

    frame_count += 1
    h, w = frame.shape[:2]
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # 1. Face Detection
    try:
        results_det = face_detection.process(rgb_frame)
    except Exception as e:
        print(f"\nFATAL ERROR in face_detection.process: {e}")
        break

    detection_info = None
    detection_flag = False
    microsleep_reason = ""

    if results_det.detections:
        face_absent_counter = 0

        detection = results_det.detections[0]
        bboxC = detection.location_data.relative_bounding_box
        x_min = int(bboxC.xmin * w)
        y_min = int(bboxC.ymin * h)
        box_width = int(bboxC.width * w)
        box_height = int(bboxC.height * h)

        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(w, x_min + box_width)
        y_max = min(h, y_min + box_height)

        face_img = frame[y_min:y_max, x_min:x_max]

        if face_img.size == 0:
            # Skip this frame if no face ROI extracted properly
            continue

        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        try:
            results_mesh = face_mesh.process(face_rgb)

            if results_mesh and results_mesh.multi_face_landmarks:
                mesh_landmarks = results_mesh.multi_face_landmarks[0].landmark

                landmarks_in_frame_coords = []
                for lm in mesh_landmarks:
                    lx_abs = (lm.x * box_width) + x_min
                    ly_abs = (lm.y * box_height) + y_min
                    landmarks_in_frame_coords.append(SimpleLandmark(x=lx_abs, y=ly_abs))

                detection_info = {
                    'bbox': (x_min, y_min, x_max, y_max),
                    'landmarks': landmarks_in_frame_coords
                }

                left_EAR = eye_aspect_ratio(landmarks_in_frame_coords, LEFT_EYE_IDX)
                right_EAR = eye_aspect_ratio(landmarks_in_frame_coords, RIGHT_EYE_IDX)
                avg_EAR = (left_EAR + right_EAR) / 2

                pitch, yaw, roll, pose_points = get_head_pose(landmarks_in_frame_coords, frame.shape)
                detection_info['pitch'] = pitch
                detection_info['pose_points'] = pose_points

                if avg_EAR < EYE_AR_THRESH:
                    microsleep_counter += 1
                else:
                    microsleep_counter = 0

                eye_drowsy = microsleep_counter >= EYE_AR_CONSEC_FRAMES
                head_nod = (pitch is not None and pitch > HEAD_NOD_ANGLE_THRESH)

                if eye_drowsy or head_nod:
                    detection_flag = True
                    microsleep_reason = "Microsleep (Eyes)" if eye_drowsy else "Head Nod (Pitch)"

            else:
                microsleep_counter = 0
        except Exception as e:
            print(f"Warning: Face mesh or head pose processing failed on frame {frame_count}: {e}")
            microsleep_counter = 0

        if detection_info:
            color = (0, 0, 255) if detection_flag else (0, 255, 0)
            (x1, y1, x2, y2) = detection_info['bbox']
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

            for lm in detection_info['landmarks']:
                cv2.circle(frame, (lm.x, lm.y), 1, (255, 0, 0), -1)

            if detection_info.get('pose_points') is not None:
                (nose_x, nose_y, end_x, end_y) = detection_info['pose_points']
                cv2.line(frame, (nose_x, nose_y), (end_x, end_y), (0, 255, 255), 3)

            pitch = detection_info.get('pitch')
            pitch_str = f"{pitch:.1f}" if pitch is not None else "N/A"
            status_text = f"EAR: {avg_EAR:.2f} | Pitch: {pitch_str} deg"
            if detection_flag:
                status_text = f"{microsleep_reason.upper()}! | " + status_text

            cv2.putText(frame, status_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

            if detection_flag:
                frame_save_path = os.path.join(save_dir,
                                               f"microsleep_frame_{microsleep_reason.replace(' ', '_')}_{frame_count:06d}.jpg")
                cv2.imwrite(frame_save_path, frame)
                saved_frames += 1
                print(f"Frame {frame_count}: >>> Drowsiness detected ({microsleep_reason})! Frame saved.")

    else:
        microsleep_counter = 0
        face_absent_counter += 1

        if face_absent_counter == FACE_ABSENT_CONSEC_FRAMES:
            filename = os.path.join(save_dir, f'absence_frame_{frame_count:06d}.jpg')
            cv2.imwrite(filename, frame)
            saved_frames += 1
            print(f"Frame {frame_count}: >>> Face absent for {FACE_ABSENT_CONSEC_FRAMES} frames! Frame saved.")

cap.release()
print(f"\nProcessing complete: {saved_frames} event frames saved in '{save_dir}'.")
