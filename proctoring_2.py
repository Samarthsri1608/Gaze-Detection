# proctoring.py
"""
Proctoring script (monocular) with head-centered gaze calibration.
Uses calibration_gui.CalibrationGUI to show fullscreen calibration points.
Run: python proctoring.py
Requirements: opencv-python, mediapipe, numpy, tkinter (standard)
"""

import cv2
import mediapipe as mp
import numpy as np
import time
import os
import csv
from datetime import datetime

from calib_gui_2 import CalibrationGUI

# ---------------- SETTINGS ----------------
CALIB_DURATION = 2.0                 # seconds per calibration point (should match GUI duration)
CALIB_SAVE_PATH = "head_calib.npz"
LOG_FILE = "proctor_log.csv"
SCREENSHOT_DIR = "proctor_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

# Gaze-out thresholds
GAZE_AWAY_SEC = 3.0                  # continuous seconds outside central zone to flag
CENTRAL_BOX = (0.15, 0.85, 0.15, 0.85)  # (xmin,xmax,ymin,ymax) in normalized screen coords

# Mediapipe settings
mp_face_mesh = mp.solutions.face_mesh
MP_MAX_FACES = 1

# Landmarks sets
LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

# Face model indices (for solvePnP)
FACE_2D_IDX = [33, 263, 1, 61, 291, 199]
FACE_3D_MODEL = np.array([
    [-30.0, -30.0,  30.0],  # left eye corner
    [ 30.0, -30.0,  30.0],  # right eye corner
    [  0.0,   0.0,  50.0],  # nose tip
    [-25.0,  30.0,  30.0],  # left mouth
    [ 25.0,  30.0,  30.0],  # right mouth
    [  0.0,  60.0,  30.0],  # chin-ish
], dtype=np.float64)

# ---------------- Helpers ----------------
def log_event(event_type, details="", frame=None):
    t = datetime.utcnow().isoformat()
    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([t, event_type, details])
    if frame is not None:
        safe_ts = t.replace(":", "-")
        fn = os.path.join(SCREENSHOT_DIR, f"{safe_ts}_{event_type}.jpg")
        cv2.imwrite(fn, frame)

# ---------------- Camera intrinsics helper ----------------
def make_camera_matrix(frame_w, frame_h, fx=None, fy=None):
    if fx is None:
        fx = frame_w
    if fy is None:
        fy = frame_w
    cx = frame_w / 2.0
    cy = frame_h / 2.0
    K = np.array([[fx, 0, cx],
                  [0, fy, cy],
                  [0,  0,  1]], dtype=np.float64)
    return K

# ---------------- Head-pose and ray functions ----------------
def estimate_head_pose(landmarks, frame, K=None):
    h, w = frame.shape[:2]
    if K is None:
        K = make_camera_matrix(w, h)
    try:
        pts2d = np.array([[landmarks.landmark[i].x * w,
                           landmarks.landmark[i].y * h] for i in FACE_2D_IDX], dtype=np.float64)
    except Exception:
        return None
    dist = np.zeros((4, 1))
    try:
        ok, rvec, tvec = cv2.solvePnP(FACE_3D_MODEL, pts2d, K, dist, flags=cv2.SOLVEPNP_EPNP)
        if not ok:
            return None
        R, _ = cv2.Rodrigues(rvec)
        t = tvec.reshape(3, 1)
        return R, t, K
    except Exception:
        return None

def pixel_ray_camera(u, v, K_inv):
    vec = np.array([u, v, 1.0], dtype=np.float64)
    r = K_inv.dot(vec)
    r = r / np.linalg.norm(r)
    return r

def compute_dir_head(landmarks, frame, R, K):
    h, w = frame.shape[:2]
    # get full landmark array (pixel coords)
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])
    # validate iris indices
    max_idx = pts.shape[0] - 1
    if max(LEFT_IRIS + RIGHT_IRIS) > max_idx:
        return None
    left = pts[LEFT_IRIS].mean(axis=0)
    right = pts[RIGHT_IRIS].mean(axis=0)
    iris = (left + right) / 2.0
    u, v = float(iris[0]), float(iris[1])
    K_inv = np.linalg.inv(K)
    r_cam = pixel_ray_camera(u, v, K_inv)   # unit ray in camera coords
    # transform to head coords: dir_head = R^T * r_cam (R maps model->camera, so transpose maps camera->model/head)
    dir_head = R.T.dot(r_cam)
    dir_head = dir_head / np.linalg.norm(dir_head)
    return dir_head

# ---------------- Head-centered calibration model ----------------
class HeadCenteredModel:
    def __init__(self):
        self.X = []   # rows: [dx,dy,dz, 1]
        self.Y = []   # rows: [sx, sy]
        self.coeff = None

    def add(self, dir_head, screen_xy):
        v = np.hstack([dir_head, 1.0])
        self.X.append(v)
        self.Y.append(screen_xy)

    def train(self):
        if len(self.X) < 4:
            raise RuntimeError("Not enough calibration samples to train (need >=4).")
        A = np.array(self.X)   # Nx4
        B = np.array(self.Y)   # Nx2
        self.coeff = np.linalg.lstsq(A, B, rcond=None)[0]

    def predict(self, dir_head):
        if self.coeff is None:
            return None
        v = np.hstack([dir_head, 1.0])
        p = v.dot(self.coeff)   # [sx, sy]
        return np.clip(p, 0.0, 1.0)

    def save(self, path=CALIB_SAVE_PATH):
        np.savez(path, coeff=self.coeff)

    def load(self, path=CALIB_SAVE_PATH):
        if not os.path.exists(path):
            return False
        data = np.load(path, allow_pickle=True)
        self.coeff = data["coeff"]
        return True

# ---------------- Blink helper (simple EAR) ----------------
def eye_aspect_ratio(eye_pts):
    # eye_pts must be Nx2 with appropriate points — we will supply 6-point arrays
    if eye_pts is None or eye_pts.shape[0] < 6:
        return 1.0
    A = np.linalg.norm(eye_pts[1] - eye_pts[5])
    B = np.linalg.norm(eye_pts[2] - eye_pts[4])
    C = np.linalg.norm(eye_pts[0] - eye_pts[3]) + 1e-6
    ear = (A + B) / (2.0 * C)
    return ear

# ---------------- Main: calibration collector + proctor loop ----------------
def collect_samples_for_point_factory(cap, face_mesh, head_model, duration=CALIB_DURATION):
    """
    Returns a function that matches the signature expected by CalibrationGUI:
      fn(name, nx, ny)
    The returned function collects `duration` seconds worth of head-centered samples
    and adds them to head_model.
    """
    def collect(name, nx, ny):
        print(f"[CALIB] Start collecting for {name} at ({nx:.2f},{ny:.2f}) for {duration}s")
        end = time.time() + duration
        samples = 0
        while time.time() < end:
            ret, frame = cap.read()
            if not ret:
                continue
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            res = face_mesh.process(rgb)
            if not res.multi_face_landmarks:
                # no reliable face -> skip
                continue
            lm = res.multi_face_landmarks[0]
            hp = estimate_head_pose(lm, frame)
            if hp is None:
                continue
            R, t, K = hp
            dir_head = compute_dir_head(lm, frame, R, K)
            if dir_head is None:
                continue
            head_model.add(dir_head, np.array([nx, ny], dtype=float))
            samples += 1
            # tiny sleep to avoid hogging CPU extremely hard
            # but we want as many samples as possible in the duration
            time.sleep(0.01)
        print(f"[CALIB] Collected {samples} samples for {name}")
    return collect

def proctor_loop(cap, face_mesh, head_model):
    print("[PROCTOR] Entering proctor loop. Press ESC to quit.")
    gaze_off_start = None
    blink_counter = 0
    total_blinks = 0
    BLINK_THRESH = 0.20
    BLINK_CONSEC_FRAMES = 3

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = face_mesh.process(rgb)
        if not res.multi_face_landmarks:
            cv2.putText(frame, "NO FACE", (30, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)
            cv2.imshow("Proctor", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:
                break
            continue

        lm = res.multi_face_landmarks[0]
        hp = estimate_head_pose(lm, frame)
        if hp is None:
            cv2.imshow("Proctor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue
        R, t, K = hp
        dir_head = compute_dir_head(lm, frame, R, K)
        if dir_head is None:
            cv2.imshow("Proctor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        pred = head_model.predict(dir_head)
        if pred is not None:
            # draw predicted gaze on frame (mapped to current frame size for visualization)
            px = int(pred[0] * frame.shape[1])
            py = int(pred[1] * frame.shape[0])
            cv2.circle(frame, (px, py), 10, (0, 255, 255), -1)

            # check central box
            xmin, xmax, ymin, ymax = CENTRAL_BOX
            if not (xmin <= pred[0] <= xmax and ymin <= pred[1] <= ymax):
                if gaze_off_start is None:
                    gaze_off_start = time.time()
                elif time.time() - gaze_off_start > GAZE_AWAY_SEC:
                    log_event("GAZE_AWAY", details=f"pred={pred}", frame=frame)
                    gaze_off_start = None  # reset to avoid flooding
            else:
                gaze_off_start = None

        # blink detection (approx)
        # try to extract approximate eye landmarks for EAR
        pts = np.array([[lm2.x * frame.shape[1], lm2.y * frame.shape[0]] for lm2 in lm.landmark])
        try:
            left_eye_pts = pts[[33,160,158,133,153,144]]
            right_eye_pts = pts[[362,385,387,263,373,380]]
            ear_l = eye_aspect_ratio(left_eye_pts)
            ear_r = eye_aspect_ratio(right_eye_pts)
            ear = (ear_l + ear_r) / 2.0
        except Exception:
            ear = 1.0

        if ear < BLINK_THRESH:
            blink_counter += 1
        else:
            if blink_counter >= BLINK_CONSEC_FRAMES:
                total_blinks += 1
            blink_counter = 0

        cv2.putText(frame, f"EAR:{ear:.2f} Blinks:{total_blinks}", (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,0), 2)

        cv2.imshow("Proctor", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == 27:
            break

    print("[PROCTOR] Exiting proctor loop.")

# ---------------- Main entry ----------------
def main():
    # Open camera
    cap = cv2.VideoCapture(0)
    time.sleep(0.3)

    # Prepare mediapipe
    with mp_face_mesh.FaceMesh(
        max_num_faces=MP_MAX_FACES,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        # head-centered calibration model
        head_model = HeadCenteredModel()

        # Try to load existing calibration if present
        if head_model.load(CALIB_SAVE_PATH):
            print("[MAIN] Loaded saved head-centered calibration.")
        else:
            print("[MAIN] No saved calibration found; starting GUI.")

            # factory function returns a function collect(name,nx,ny) that uses cap & face_mesh
            collector = collect_samples_for_point_factory(cap, face_mesh, head_model, duration=CALIB_DURATION)

            # Launch fullscreen GUI which will call the collector synchronously
            gui = CalibrationGUI(on_point_callback=collector, duration_per_point=CALIB_DURATION)
            gui.start()  # blocks until GUI finishes

            # After GUI returns, train model
            try:
                print("[MAIN] Training head-centered calibration model...")
                head_model.train()
                head_model.save(CALIB_SAVE_PATH)
                print("[MAIN] Calibration trained and saved.")
            except Exception as e:
                print("[MAIN] Calibration failed:", e)
                # If training failed, exit gracefully
                cap.release()
                cv2.destroyAllWindows()
                return

        # Start proctor loop
        proctor_loop(cap, face_mesh, head_model)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
