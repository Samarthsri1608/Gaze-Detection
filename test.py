# proctoring.py
import cv2
import mediapipe as mp
import numpy as np
import time
import csv
import os
from datetime import datetime

from calibration_gui import CalibrationGUI


# ------------------ GLOBAL SETTINGS ------------------
CALIBRATION_DURATION = 2       # seconds per point
LOG_FILE = "proctor_log.csv"
SCREENSHOT_DIR = "proctor_screenshots"
os.makedirs(SCREENSHOT_DIR, exist_ok=True)

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

LEFT_EYE = [33, 133, 159, 145]
RIGHT_EYE = [263, 362, 386, 374]


# ------------------ Helper Classes ------------------
class EMAFilter:
    def __init__(self, alpha=0.2):
        self.alpha = alpha
        self.value = None

    def update(self, x):
        x = np.array(x, dtype=float)
        if self.value is None:
            self.value = x
        else:
            self.value = self.alpha * x + (1 - self.alpha) * self.value
        return self.value


class CalibrationModel:
    def __init__(self):
        self.eye_samples = []
        self.screen_samples = []
        self.coeff = None

    def add_sample(self, eye_pos, screen_pos):
        self.eye_samples.append(eye_pos)
        self.screen_samples.append(screen_pos)

    def train(self):
        A = np.array(self.eye_samples)
        B = np.array(self.screen_samples)

        A_ext = np.hstack([A, np.ones((A.shape[0], 1))])
        self.coeff = np.linalg.lstsq(A_ext, B, rcond=None)[0]

    def predict(self, eye_pos):
        if self.coeff is None:
            return None
        eye = np.array([eye_pos[0], eye_pos[1], 1.0])
        return eye @ self.coeff


calibration = CalibrationModel()
ema = EMAFilter(alpha=0.25)


# ------------------ Logging ------------------
def log_event(event_type, details="", frame=None):
    t = datetime.utcnow().isoformat()

    with open(LOG_FILE, "a", newline="") as f:
        w = csv.writer(f)
        w.writerow([t, event_type, details])

    if frame is not None:
        filename = f"{SCREENSHOT_DIR}/{t.replace(':','-')}_{event_type}.jpg"
        cv2.imwrite(filename, frame)


# ------------------ Eye Feature Extraction ------------------
def extract_eye_position(landmarks, frame):
    """Returns relative iris position (0–1 range in eye box)."""
    h, w = frame.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])

    try:
        left_iris = pts[LEFT_IRIS].mean(axis=0)
        right_iris = pts[RIGHT_IRIS].mean(axis=0)
        iris = (left_iris + right_iris) / 2

        eye_pts = pts[LEFT_EYE + RIGHT_EYE]
        x_min, y_min = eye_pts.min(axis=0)
        x_max, y_max = eye_pts.max(axis=0)

        rel_x = (iris[0] - x_min) / (x_max - x_min + 1e-5)
        rel_y = (iris[1] - y_min) / (y_max - y_min + 1e-5)
        return np.array([rel_x, rel_y])

    except:
        return None


# ------------------ Calibration Sample Collection ------------------
def collect_samples_for_point(name, nx, ny):
    """Collects eye samples for 2 seconds for this calibration dot."""
    print(f"[CALIB] Collecting samples for {name}")

    end_time = time.time() + CALIBRATION_DURATION

    cap = collect_samples_for_point.cap
    mesh = collect_samples_for_point.mesh

    while time.time() < end_time:
        ret, frame = cap.read()
        if not ret:
            continue

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = mesh.process(rgb)

        if result.multi_face_landmarks:
            eye = extract_eye_position(result.multi_face_landmarks[0], frame)
            if eye is not None:
                calibration.add_sample(eye, np.array([nx, ny]))


# ------------------ MAIN PROCTOR LOOP ------------------
def proctor_loop(cap, face_mesh):
    print("\n=== Proctoring Started ===\n")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = face_mesh.process(rgb)

        if not result.multi_face_landmarks:
            cv2.putText(frame, "NO FACE DETECTED!", (30, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            log_event("FACE_MISSING", "No face detected", frame)
            cv2.imshow("Proctor", frame)
            if cv2.waitKey(1) & 0xFF == 27:
                break
            continue

        # Use primary face
        lm = result.multi_face_landmarks[0]
        eye = extract_eye_position(lm, frame)

        if eye is not None:
            pred = calibration.predict(eye)
            if pred is not None:
                pred = ema.update(pred)
                px = int(pred[0] * frame.shape[1])
                py = int(pred[1] * frame.shape[0])
                cv2.circle(frame, (px, py), 12, (0, 255, 255), -1)

        cv2.imshow("Proctor", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break


# ------------------ MAIN ENTRY ------------------
def main():
    cap = cv2.VideoCapture(0)
    time.sleep(0.5)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        # Pass cap & face_mesh to callback for calibration
        collect_samples_for_point.cap = cap
        collect_samples_for_point.mesh = face_mesh

        # ----------- Start Calibration GUI ------------
        gui = CalibrationGUI(on_point_callback=collect_samples_for_point)
        gui.start()

        # ----------- Train Calibration ----------------
        print("[CALIB] Training model...")
        calibration.train()
        print("[CALIB] Done!")

        # ----------- Start Proctoring -----------------
        proctor_loop(cap, face_mesh)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
