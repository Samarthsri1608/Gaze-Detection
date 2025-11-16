# proctor_gaze.py
# Augmented version of your test.py for basic exam-proctoring flags + logging.
# Requires: opencv-python, mediapipe, numpy
# Run: python proctor_gaze.py

import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from collections import deque
from datetime import datetime

mp_face_mesh = mp.solutions.face_mesh

# landmark sets (assumes refine_landmarks=True available)
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]
LEFT_EYE_LANDMARKS = [33, 133, 159, 145]
RIGHT_EYE_LANDMARKS = [263, 362, 386, 374]
FACE_LANDMARK_COUNT_MIN = 478  # approximate when refine landmarks present

# ----------------- Parameters -----------------
SCREEN_W = 1280
SCREEN_H = 720
CALIB_SAMPLE_TIME = 2.0   # seconds per calibration dot
EMA_ALPHA = 0.25
MIN_FACE_AREA = 3000      # if face bbox area smaller than this, treat as far/occluded
SUSPICIOUS_GAZE_SEC = 3.0    # seconds of continuous gaze-away to flag
MULTI_FACE_FLAG = True
MAX_ALLOWED_FACES = 1
BLINK_THRESHOLD = 0.20    # eye aspect ratio threshold for closed eye (heuristic)
BLINK_CONSEC_FRAMES = 3   # consecutive frames count to register a blink
SCREENSHOT_DIR = "proctor_screenshots"
LOG_FILE = "proctor_log.csv"

os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# ----------------- Utilities -----------------
class EMAFilter:
    def __init__(self, alpha=EMA_ALPHA):
        self.alpha = alpha
        self.v = None
    def update(self, x):
        x = np.array(x, dtype=float)
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1-self.alpha) * self.v
        return self.v

ema_filter = EMAFilter()

class CalibrationModel:
    def __init__(self):
        self.calib_eye = []
        self.calib_screen = []
        self.coeff = None
    def add_sample(self, eye_xy, screen_xy_norm):
        self.calib_eye.append(eye_xy)
        self.calib_screen.append(screen_xy_norm)
    def train(self):
        if len(self.calib_eye) < 3:
            print("[Calibration] Warning: fewer than 3 samples.")
        A = np.array(self.calib_eye)
        B = np.array(self.calib_screen)
        A_ext = np.hstack([A, np.ones((A.shape[0],1))])
        self.coeff = np.linalg.lstsq(A_ext, B, rcond=None)[0]
    def predict(self, eye_xy):
        if self.coeff is None:
            return None
        vec = np.array([eye_xy[0], eye_xy[1], 1.0])
        return vec @ self.coeff
    def save(self, path="calib.npz"):
        np.savez(path, coeff=self.coeff)
    def load(self, path="calib.npz"):
        if os.path.exists(path):
            data = np.load(path, allow_pickle=True)
            self.coeff = data["coeff"]
            return True
        return False

calibration = CalibrationModel()

# Logging helper
def log_event(event_type, details="", frame=None):
    t = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([t, event_type, details])
    if frame is not None:
        fname = f"{SCREENSHOT_DIR}/{t.replace(':','-')}_{event_type}.jpg"
        cv2.imwrite(fname, frame)

# --- Eye-relative extraction robustified ---
def extract_relative_eye_pos(landmarks, frame):
    # landmarks: mediapipe LandmarkList
    h, w = frame.shape[:2]
    try:
        pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
    except Exception:
        return None
    # Check if landmarks count sufficient for iris indices
    if pts.shape[0] <= max(LEFT_IRIS+RIGHT_IRIS+LEFT_EYE_LANDMARKS+RIGHT_EYE_LANDMARKS):
        return None
    left_iris = pts[LEFT_IRIS].mean(axis=0)
    right_iris = pts[RIGHT_IRIS].mean(axis=0)
    left_eye_pts = pts[LEFT_EYE_LANDMARKS]
    right_eye_pts = pts[RIGHT_EYE_LANDMARKS]
    eye_all = np.vstack([left_eye_pts, right_eye_pts])
    x_min, y_min = eye_all.min(axis=0)
    x_max, y_max = eye_all.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    if width <= 2 or height <= 2:
        return None
    iris = (left_iris + right_iris) / 2.0
    rel_x = (iris[0] - x_min) / width
    rel_y = (iris[1] - y_min) / height
    rel_x = float(np.clip(rel_x, 0.0, 1.0))
    rel_y = float(np.clip(rel_y, 0.0, 1.0))
    return np.array([rel_x, rel_y]), (x_min,y_min,x_max,y_max), pts

# Head pose using 6 points + solvePnP (rough)
FACE_2D_IDX = [33, 263, 1, 61, 291, 199]  # approximate: left eye corner, right eye corner, nose tip, mouth corners, chin-ish
FACE_3D_MODEL = np.array([
    [-30.0,  -30.0,   30.0],   # left eye corner (approx)
    [ 30.0,  -30.0,   30.0],   # right eye corner
    [  0.0,    0.0,   50.0],   # nose tip
    [-25.0,   30.0,   30.0],   # left mouth
    [ 25.0,   30.0,   30.0],   # right mouth
    [  0.0,   60.0,   30.0],   # chin-ish
], dtype=np.float64)

def estimate_head_pose(pts2d, frame):
    h, w = frame.shape[:2]
    image_pts = np.array([pts2d[i] for i in FACE_2D_IDX], dtype=np.float64)
    focal_length = w
    center = (w/2, h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]], [0, focal_length, center[1]], [0,0,1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))
    try:
        ok, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, image_pts, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        rot_mat, _ = cv2.Rodrigues(rvec)
        # compute yaw from rot_mat
        yaw = np.arctan2(rot_mat[1,0], rot_mat[0,0]) * 180.0/np.pi
        pitch = np.arcsin(-rot_mat[2,0]) * 180.0/np.pi
        roll = np.arctan2(rot_mat[2,1], rot_mat[2,2]) * 180.0/np.pi
        return {"yaw": yaw, "pitch": pitch, "roll": roll}
    except Exception:
        return None

# Blink / eye aspect ratio using landmarks around eye (heuristic)
def eye_aspect_ratio(eye_pts):
    # eye_pts: Nx2 points for the eye contour approximate; use distance ratios
    # We'll select vertical pairs vs horizontal distance
    if eye_pts.shape[0] < 6:
        return 1.0
    A = np.linalg.norm(eye_pts[1]-eye_pts[5])
    B = np.linalg.norm(eye_pts[2]-eye_pts[4])
    C = np.linalg.norm(eye_pts[0]-eye_pts[3])
    if C == 0:
        return 1.0
    ear = (A+B) / (2.0*C)
    return ear

# ----------------- Calibration UI -----------------
CALIB_POINTS = [
    ("CENTER", 0.5, 0.5),
    ("LEFT",   0.12, 0.5),
    ("RIGHT",  0.88, 0.5),
    ("TOP",    0.5, 0.18),
    ("BOTTOM", 0.5, 0.82),
]

def run_calibration(cap, face_mesh, sample_time=CALIB_SAMPLE_TIME):
    print("[Calibration] Keep your head reasonably still. Collecting samples...")
    for name, nx, ny in CALIB_POINTS:
        cx = int(nx * SCREEN_W)
        cy = int(ny * SCREEN_H)
        samples = []
        start = time.time()
        while time.time() - start < sample_time:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)
            # show marker centered on frame (scaled to frame)
            sx = int(nx * frame.shape[1])
            sy = int(ny * frame.shape[0])
            cv2.circle(frame, (sx, sy), 18, (0,0,255), -1)
            cv2.putText(frame, f"Look at {name}", (30,30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)
            if results.multi_face_landmarks:
                out = extract_relative_eye_pos(results.multi_face_landmarks[0], frame)
                if out is not None:
                    rel, _, _ = out
                    samples.append(rel)
            cv2.imshow("Calibration", frame)
            if (cv2.waitKey(1) & 0xFF) == 27:
                break
        if samples:
            mean_eye = np.mean(samples, axis=0)
            calibration.add_sample(mean_eye, np.array([nx, ny]))
            print(f"[Calibration] Collected {len(samples)} samples for {name}")
        else:
            print(f"[Calibration] No samples for {name} — try again or improve lighting.")
    if calibration.calib_eye:
        calibration.train()
        calibration.save("calib.npz")
        print("[Calibration] Training finished and saved.")
    cv2.destroyWindow("Calibration")

# ----------------- Main proctoring loop -----------------
def proctor_loop():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)
    # optional: try to load calibration if exists
    if calibration.load("calib.npz"):
        print("[Calibration] Loaded saved calibration.")
    with mp_face_mesh.FaceMesh(max_num_faces=4, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        # run calibration if not loaded
        if calibration.coeff is None:
            run_calibration(cap, face_mesh)

        gaze_off_start = None
        face_missing_start = None
        blink_counter = 0
        total_blinks = 0
        frame_idx = 0

        # initialize log file header
        if not os.path.exists(LOG_FILE):
            with open(LOG_FILE, "w", newline="") as f:
                w = csv.writer(f)
                w.writerow(["utc_ts","event","details"])

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # default statuses
            faces_detected = 0
            suspicious_reasons = []

            if not results.multi_face_landmarks:
                faces_detected = 0
                # face missing
                if face_missing_start is None:
                    face_missing_start = time.time()
                elif time.time() - face_missing_start > 2.0:
                    suspicious_reasons.append("FACE_MISSING")
                    log_event("FACE_MISSING", details="No face detected", frame=frame)
                    face_missing_start = time.time()  # reset to avoid flooding logs
                cv2.putText(frame, "No face detected", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2)
            else:
                face_missing_start = None
                faces_detected = len(results.multi_face_landmarks)
                # detect multiple faces
                if faces_detected > MAX_ALLOWED_FACES:
                    suspicious_reasons.append("MULTI_FACE")
                    log_event("MULTI_FACE", details=f"{faces_detected} faces", frame=frame)

                # pick first face (primary)
                face_landmarks = results.multi_face_landmarks[0]
                out = extract_relative_eye_pos(face_landmarks, frame)
                if out is None:
                    cv2.putText(frame, "Eye landmarks not reliable", (30,90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255),2)
                else:
                    rel, eye_box, pts = out
                    # map to screen using calibration (predict returns normalized coords)
                    pred_norm = calibration.predict(rel)
                    if pred_norm is not None:
                        pred_norm = ema_filter.update(pred_norm)
                        # map to frame coords for visualization
                        sx = int(pred_norm[0] * frame.shape[1])
                        sy = int(pred_norm[1] * frame.shape[0])
                        cv2.circle(frame, (sx, sy), 10, (0,255,255), -1)

                        # determine gaze-away (if predicted normalized location is outside central region)
                        # customizable thresholds:
                        if pred_norm[0] < 0.05 or pred_norm[0] > 0.95 or pred_norm[1] < 0.05 or pred_norm[1] > 0.95:
                            # outside extreme edges -> immediate flag
                            suspicious_reasons.append("GAZE_EXTREME")
                            log_event("GAZE_EXTREME", details=str(pred_norm), frame=frame)
                        else:
                            # consider "gaze-away" when gaze points outside a central rectangle for longer than SUSPICIOUS_GAZE_SEC
                            cx_min, cx_max = 0.15, 0.85
                            cy_min, cy_max = 0.15, 0.85
                            if not (cx_min <= pred_norm[0] <= cx_max and cy_min <= pred_norm[1] <= cy_max):
                                if gaze_off_start is None:
                                    gaze_off_start = time.time()
                                elif time.time() - gaze_off_start > SUSPICIOUS_GAZE_SEC:
                                    suspicious_reasons.append("GAZE_AWAY")
                                    log_event("GAZE_AWAY", details=f"pred_norm={pred_norm}", frame=frame)
                                    gaze_off_start = None
                            else:
                                gaze_off_start = None

                    # head-pose estimate
                    hp = estimate_head_pose(pts, frame)
                    if hp is not None:
                        # flag if yaw or pitch is large
                        if abs(hp['yaw']) > 30 or abs(hp['pitch']) > 25:
                            suspicious_reasons.append("HEAD_TURN")
                            log_event("HEAD_TURN", details=str(hp), frame=frame)
                        # overlay
                        cv2.putText(frame, f"yaw:{hp['yaw']:.1f}", (30,120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0),2)

                    # blink detection: compute EAR for left and right eye approximately
                    left_eye = pts[[33,160,158,133,153,144]] if pts.shape[0] > 160 else None
                    right_eye = pts[[362,385,387,263,373,380]] if pts.shape[0] > 385 else None
                    ear_l = eye_aspect_ratio(left_eye) if left_eye is not None else 1.0
                    ear_r = eye_aspect_ratio(right_eye) if right_eye is not None else 1.0
                    ear = (ear_l + ear_r) / 2.0
                    if ear < BLINK_THRESHOLD:
                        blink_counter += 1
                    else:
                        if blink_counter >= BLINK_CONSEC_FRAMES:
                            total_blinks += 1
                        blink_counter = 0
                    cv2.putText(frame, f"EAR:{ear:.2f} Blinks:{total_blinks}", (30,150), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0),2)

            # Visualization: face count
            cv2.putText(frame, f"Faces:{faces_detected}", (frame.shape[1]-200,30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255),2)

            # Suspicious reasons overlay
            if suspicious_reasons:
                cv2.putText(frame, "FLAG: " + ",".join(set(suspicious_reasons)), (30, frame.shape[0]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,0,255),2)

            cv2.imshow("Proctor Monitor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
            if key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    proctor_loop()
