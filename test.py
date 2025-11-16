import cv2
import mediapipe as mp
import numpy as np
import time
from collections import deque

mp_face_mesh = mp.solutions.face_mesh

LEFT_IRIS = [468, 469, 470, 471]
RIGHT_IRIS = [473, 474, 475, 476]

LEFT_EYE_LANDMARKS = [33, 133, 159, 145]
RIGHT_EYE_LANDMARKS = [263, 362, 386, 374]

# ---------- SMOOTHING ----------
class EMAFilter:
    def __init__(self, alpha=0.3):
        self.alpha = alpha
        self.v = None

    def update(self, x):
        if self.v is None:
            self.v = x
        else:
            self.v = self.alpha * x + (1 - self.alpha) * self.v
        return self.v

ema_filter = EMAFilter(alpha=0.25)

# ---------- GAZE MAPPING MODEL ----------
class CalibrationModel:
    def __init__(self):
        self.calib_eye = []
        self.calib_screen = []
        self.coeff = None

    def add_sample(self, eye_xy, screen_xy):
        self.calib_eye.append(eye_xy)
        self.calib_screen.append(screen_xy)

    def train(self):
        A = np.array(self.calib_eye)    # N×2
        B = np.array(self.calib_screen) # N×2

        # Add bias column → N × 3
        A_ext = np.hstack([A, np.ones((A.shape[0], 1))])

        # Solve via least squares: A_ext * coeff = B
        self.coeff = np.linalg.lstsq(A_ext, B, rcond=None)[0]

    def predict(self, eye_xy):
        if self.coeff is None:
            return None

        x, y = eye_xy
        vec = np.array([x, y, 1.0])
        return vec @ self.coeff  # returns (x_screen, y_screen)

calibration = CalibrationModel()

# ---------- CALIBRATION UI ----------
CALIB_POINTS = [
    ("CENTER", 0.5, 0.5),
    ("LEFT",   0.15, 0.5),
    ("RIGHT",  0.85, 0.5),
    ("TOP",    0.5, 0.2),
    ("BOTTOM", 0.5, 0.8),
]

def run_calibration(screen_w, screen_h, cap, face_mesh):
    print("\n--- Starting Calibration ---")

    for name, nx, ny in CALIB_POINTS:
        cx = int(nx * screen_w)
        cy = int(ny * screen_h)

        samples = []

        t_start = time.time()
        while time.time() - t_start < 1.5:   # collect for 1.5 seconds
            ret, frame = cap.read()
            if not ret:
                continue

            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            # Draw calibration dot
            cv2.circle(frame, (cx, cy), 18, (0,0,255), -1)
            cv2.putText(frame, f"Look at {name}", (30,30),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255), 2)

            if results.multi_face_landmarks:
                eye_pos = extract_relative_eye_pos(results.multi_face_landmarks[0], frame)
                if eye_pos is not None:
                    samples.append(eye_pos)

            cv2.imshow("Calibration", frame)
            if cv2.waitKey(1) == 27:
                break

        if samples:
            mean_eye = np.mean(samples, axis=0)
            calibration.add_sample(mean_eye, np.array([cx, cy]))

    calibration.train()
    cv2.destroyWindow("Calibration")
    print("Calibration finished.\n")

# ---------- RELATIVE EYE POSITION ----------
def extract_relative_eye_pos(landmarks, frame):
    h, w = frame.shape[:2]
    pts = np.array([[lm.x * w, lm.y * h] for lm in landmarks.landmark])

    # iris centers
    left_iris = pts[LEFT_IRIS].mean(axis=0)
    right_iris = pts[RIGHT_IRIS].mean(axis=0)

    # eye region box
    left_eye_pts = pts[LEFT_EYE_LANDMARKS]
    right_eye_pts = pts[RIGHT_EYE_LANDMARKS]

    # combine both eyes
    iris = (left_iris + right_iris) / 2
    eye_all = np.vstack([left_eye_pts, right_eye_pts])

    x_min, y_min = eye_all.min(axis=0)
    x_max, y_max = eye_all.max(axis=0)

    # normalized pupil location
    rel_x = (iris[0] - x_min) / (x_max - x_min)
    rel_y = (iris[1] - y_min) / (y_max - y_min)

    return np.array([rel_x, rel_y])


# ---------- MAIN LOOP ----------
cap = cv2.VideoCapture(0)
screen_w = 1280
screen_h = 720

with mp_face_mesh.FaceMesh(max_num_faces=1, refine_landmarks=True) as face_mesh:

    # --- CALIBRATION ---
    run_calibration(screen_w, screen_h, cap, face_mesh)

    # --- MAIN GAZE LOOP ---
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            eye_pos = extract_relative_eye_pos(results.multi_face_landmarks[0], frame)
            if eye_pos is not None:
                # map via calibration model
                screen_xy = calibration.predict(eye_pos)
                if screen_xy is not None:
                    screen_xy = ema_filter.update(screen_xy)

                    sx, sy = map(int, screen_xy)

                    # draw predicted point
                    cv2.circle(frame, (sx, sy), 10, (0,255,255), -1)

                # debug text
                cv2.putText(frame, f"rel: {eye_pos[0]:.2f},{eye_pos[1]:.2f}",
                            (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7,(0,255,0),2)

        cv2.imshow("Gaze Tracking", frame)
        if cv2.waitKey(1) == 27:
            break

cap.release()
cv2.destroyAllWindows()
