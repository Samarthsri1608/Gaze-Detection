import cv2, mediapipe as mp, numpy as np, pandas as pd, time
import pyautogui
from sklearn.linear_model import LinearRegression

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(0)

screen_w, screen_h = pyautogui.size()  

calibration_points = [
    ("CENTER", (screen_w//2, screen_h//2)),
    ("TOP_LEFT", (0, 0)),
    ("TOP_RIGHT", (screen_w, 0)),
    ("BOTTOM_LEFT", (0, screen_h)),
    ("BOTTOM_RIGHT", (screen_w, screen_h))
]

X_data, y_data = [], []
model_x, model_y = LinearRegression(), LinearRegression()

def extract_features(landmarks):
    # Simple features: iris center + head yaw/pitch
    iris_left = np.array([landmarks[473].x, landmarks[473].y])
    iris_right = np.array([landmarks[468].x, landmarks[468].y])
    iris_center = (iris_left + iris_right) / 2
    return iris_center

def estimate_head_pose(landmarks, w, h):
    model_points = np.array([
        (0.0, 0.0, 0.0), (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0), (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0), (150.0, -150.0, -125.0)
    ], dtype=np.float64)
    image_points = np.array([
        (landmarks[1].x * w, landmarks[1].y * h),
        (landmarks[152].x * w, landmarks[152].y * h),
        (landmarks[263].x * w, landmarks[263].y * h),
        (landmarks[33].x * w, landmarks[33].y * h),
        (landmarks[287].x * w, landmarks[287].y * h),
        (landmarks[57].x * w, landmarks[57].y * h)
    ], dtype=np.float64)
    focal_length = w
    cam_matrix = np.array([[focal_length, 0, w/2],
                           [0, focal_length, h/2],
                           [0, 0, 1]], dtype=np.float64)
    dist_coeffs = np.zeros((4,1))
    success, rot_vec, trans_vec = cv2.solvePnP(
        model_points, image_points, cam_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    rmat, _ = cv2.Rodrigues(rot_vec)
    proj = np.hstack((rmat, trans_vec))
    euler, _ = cv2.decomposeProjectionMatrix(proj)[:2]
    return euler[1][0], euler[0][0]  # yaw, pitch

# --- Calibration Phase ---
print("Starting calibration...")
for name, (tx, ty) in calibration_points:
    print(f"Look at the {name} point on screen...")
    time.sleep(2)
    samples = []
    while len(samples) < 10:
        success, frame = cap.read()
        if not success: continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            iris = extract_features(lm)
            yaw, pitch = estimate_head_pose(lm, w, h)
            samples.append([iris[0], iris[1], yaw, pitch])
        cv2.putText(frame, f"Look at {name}", (50,50), cv2.FONT_HERSHEY_SIMPLEX,1,(0,255,0),2)
        cv2.imshow("Calibrating...", frame)
        if cv2.waitKey(1) & 0xFF == 27: break
    avg_features = np.mean(samples, axis=0)
    X_data.append(avg_features)
    y_data.append([tx, ty])

X_data, y_data = np.array(X_data), np.array(y_data)
model_x.fit(X_data, y_data[:,0])
model_y.fit(X_data, y_data[:,1])
print("Calibration complete")

# --- Live Tracking ---
log = []
warning_start = None
while True:
    success, frame = cap.read()
    if not success: break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        iris = extract_features(lm)
        yaw, pitch = estimate_head_pose(lm, w, h)
        feat = np.array([[iris[0], iris[1], yaw, pitch]])
        pred_x = model_x.predict(feat)[0]
        pred_y = model_y.predict(feat)[0]

        # Draw predicted point on a mini-screen box
        cv2.rectangle(frame, (50, 50), (350, 250), (255,255,255), 1)
        cv2.circle(frame, (int(50+pred_x/screen_w*300), int(50+pred_y/screen_h*200)), 5, (0,255,0), -1)

        # Check if gaze out of screen bounds
        if not (0 <= pred_x <= screen_w and 0 <= pred_y <= screen_h):
            if warning_start is None:
                warning_start = time.time()
            elif time.time() - warning_start > 1.5:
                cv2.putText(frame, "⚠ OFF-SCREEN!", (30, 120),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0,0,255), 3)
                log.append([time.strftime("%H:%M:%S"), pred_x, pred_y, "Off-screen"])
        else:
            warning_start = None

    cv2.imshow("Gaze Monitor (Calibrated)", frame)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

pd.DataFrame(log, columns=["time","x","y","event"]).to_csv("gaze_log.csv", index=False)
print("Logs saved to gaze_log.csv")
