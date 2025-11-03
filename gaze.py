import cv2, mediapipe as mp, numpy as np, pandas as pd, time
import tkinter as tk
from sklearn.linear_model import LinearRegression

# Get screen size using tkinter
root = tk.Tk()
screen_w = root.winfo_screenwidth()
screen_h = root.winfo_screenheight()
root.destroy()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(refine_landmarks=True)
cap = cv2.VideoCapture(2)

calibration_points = [
    ("CENTER", (screen_w//2, screen_h//2)),
    ("TOP_LEFT", (0, 0)),
    ("TOP_RIGHT", (screen_w, 0)),
    ("BOTTOM_LEFT", (0, screen_h)),
    ("BOTTOM_RIGHT", (screen_w, screen_h))
]

class GazeTracker:
    def __init__(self):
        self.model_x = LinearRegression()
        self.model_y = LinearRegression()
        self.reference_features = None
        self.reference_position = None
        self.calibration_data = []
        self.buffer_size = 5
        self.x_buffer = []
        self.y_buffer = []
        self.position_history = []
        self.last_recalibration = time.time()
        self.needs_recalibration = False

    def get_face_orientation(self, landmarks):
        # Get face plane using key points
        nose_bridge = np.array([landmarks[168].x, landmarks[168].y, landmarks[168].z])
        left_temple = np.array([landmarks[447].x, landmarks[447].y, landmarks[447].z])
        right_temple = np.array([landmarks[227].x, landmarks[227].y, landmarks[227].z])
        
        # Calculate face normal vector
        face_vector1 = left_temple - nose_bridge
        face_vector2 = right_temple - nose_bridge
        face_normal = np.cross(face_vector1, face_vector2)
        face_normal = face_normal / (np.linalg.norm(face_normal) + 1e-6)
        
        return face_normal

    def extract_features(self, landmarks):
        # Get face orientation for angle compensation
        face_normal = self.get_face_orientation(landmarks)
        
        # Eye landmarks in 3D
        iris_left = np.array([landmarks[473].x, landmarks[473].y, landmarks[473].z])
        iris_right = np.array([landmarks[468].x, landmarks[468].y, landmarks[468].z])
        left_eye_outer = np.array([landmarks[33].x, landmarks[33].y, landmarks[33].z])
        left_eye_inner = np.array([landmarks[133].x, landmarks[133].y, landmarks[133].z])
        right_eye_inner = np.array([landmarks[362].x, landmarks[362].y, landmarks[362].z])
        right_eye_outer = np.array([landmarks[263].x, landmarks[263].y, landmarks[263].z])
        
        # Calculate eye vectors in face space
        left_eye_center = (left_eye_inner + left_eye_outer) / 2
        right_eye_center = (right_eye_inner + right_eye_outer) / 2
        left_eye_vector = iris_left - left_eye_center
        right_eye_vector = iris_right - right_eye_center
        
        # Normalize vectors
        left_eye_vector = left_eye_vector / (np.linalg.norm(left_eye_vector) + 1e-6)
        right_eye_vector = right_eye_vector / (np.linalg.norm(right_eye_vector) + 1e-6)
        
        # Project onto face plane
        left_eye_proj = left_eye_vector - np.dot(left_eye_vector, face_normal) * face_normal
        right_eye_proj = right_eye_vector - np.dot(right_eye_vector, face_normal) * face_normal
        
        # Normalize projected vectors
        left_eye_proj = left_eye_proj / (np.linalg.norm(left_eye_proj) + 1e-6)
        right_eye_proj = right_eye_proj / (np.linalg.norm(right_eye_proj) + 1e-6)
        
        # Combine features
        gaze_vector = (left_eye_proj + right_eye_proj) / 2
        features = np.concatenate([
            gaze_vector,
            face_normal,
            [(landmarks[6].x + landmarks[197].x) / 2],  # Nose bridge x
            [(landmarks[6].y + landmarks[197].y) / 2]   # Nose bridge y
        ])
        
        return features
        
    def calibrate(self, samples):
        X = np.array([sample[0] for sample in samples])
        y_x = np.array([sample[1][0] for sample in samples])
        y_y = np.array([sample[1][1] for sample in samples])
        
        self.model_x.fit(X, y_x)
        self.model_y.fit(X, y_y)
        self.calibration_data = samples
        self.needs_recalibration = False
        self.last_recalibration = time.time()
    
    def predict(self, features):
        if self.reference_features is None:
            self.reference_features = features
            return None
            
        # Calculate relative changes from reference
        delta_features = features - self.reference_features
        
        # Get raw predictions
        raw_x = self.model_x.predict(delta_features.reshape(1, -1))[0]
        raw_y = self.model_y.predict(delta_features.reshape(1, -1))[0]
        
        # Apply smoothing
        self.x_buffer.append(raw_x)
        self.y_buffer.append(raw_y)
        if len(self.x_buffer) > self.buffer_size:
            self.x_buffer.pop(0)
            self.y_buffer.pop(0)
        
        # Weighted average (recent predictions have more weight)
        weights = np.linspace(0.5, 1.0, len(self.x_buffer))
        weights = weights / np.sum(weights)
        smooth_x = np.average(self.x_buffer, weights=weights)
        smooth_y = np.average(self.y_buffer, weights=weights)
        
        # Store for stability checking
        self.position_history.append([smooth_x, smooth_y])
        if len(self.position_history) > 30:
            self.position_history.pop(0)
            
        # Check if recalibration is needed
        if len(self.position_history) >= 30:
            recent_positions = np.array(self.position_history[-30:])
            position_std = np.std(recent_positions, axis=0)
            
            # If high variance in positions and it's been a while since last recalibration
            if np.mean(position_std) > 100 and time.time() - self.last_recalibration > 10:
                self.needs_recalibration = True
        
        return smooth_x, smooth_y

# Initialize tracker and calibration data
tracker = GazeTracker()
calibration_samples = []

# --- Calibration Phase ---
print("Starting calibration...")

# Create full screen window
cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# Create a black background image
calib_display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)

for name, (tx, ty) in calibration_points:
    print(f"Look at the {name} point on screen...")
    time.sleep(2)
    samples = []
    stable_count = 0
    prev_features = None
    
    while len(samples) < 20:  # Increased sample size
        success, frame = cap.read()
        if not success: continue
        h, w = frame.shape[:2]
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            lm = results.multi_face_landmarks[0].landmark
            
            features = tracker.extract_features(lm)
            
            # Check stability
            if prev_features is not None:
                movement = np.mean(np.abs(features - prev_features))
                if movement < 0.01:  # Threshold for stability
                    stable_count += 1
                else:
                    stable_count = 0
            prev_features = features
            
            if stable_count >= 3:  # Only add sample when gaze is stable
                samples.append([features, (tx, ty)])
                stable_count = 0
        
        # Create calibration display
        calib_display.fill(0)  # Clear the display
        
        # Draw calibration point (larger and more visible)
        cv2.circle(calib_display, (tx, ty), 20, (0, 0, 255), -1)  # Inner red circle
        cv2.circle(calib_display, (tx, ty), 25, (255, 255, 255), 2)  # Outer white ring
        
        # Add text overlay
        font_scale = screen_h / 1080  # Scale text based on screen height
        cv2.putText(calib_display, f"Look at the {name} point", 
                   (50, 50), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, (255, 255, 255), 2)
        cv2.putText(calib_display, f"Samples: {len(samples)}/20", 
                   (50, 100), cv2.FONT_HERSHEY_DUPLEX, 
                   font_scale, (255, 255, 255), 2)
        
        # Add webcam feed in corner for monitoring
        webcam_scale = 0.2
        webcam_h = int(h * webcam_scale)
        webcam_w = int(w * webcam_scale)
        frame_small = cv2.resize(frame, (webcam_w, webcam_h))
        calib_display[50:50+webcam_h, screen_w-webcam_w-50:screen_w-50] = frame_small
        
        cv2.imshow("Calibration", calib_display)
        if cv2.waitKey(1) & 0xFF == 27: break
    calibration_samples.extend(samples)

# Initial calibration
tracker.calibrate(calibration_samples)
print("Calibration complete")

# Close calibration window and create tracking window
cv2.destroyWindow("Calibration")
cv2.namedWindow("Gaze Monitor", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("Gaze Monitor", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

# --- Live Tracking ---
log = []
warning_start = None
tracking_display = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
recalibration_points = []

while True:
    success, frame = cap.read()
    if not success: break
    h, w = frame.shape[:2]
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(rgb)

    # Clear display and add webcam feed
    tracking_display.fill(0)
    webcam_scale = 0.2
    webcam_h = int(h * webcam_scale)
    webcam_w = int(w * webcam_scale)
    frame_small = cv2.resize(frame, (webcam_w, webcam_h))
    tracking_display[50:50+webcam_h, screen_w-webcam_w-50:screen_w-50] = frame_small

    # Draw screen bounds
    cv2.rectangle(tracking_display, (0, 0), (screen_w-1, screen_h-1), (100, 100, 100), 2)

    if results.multi_face_landmarks:
        lm = results.multi_face_landmarks[0].landmark
        features = tracker.extract_features(lm)
        prediction = tracker.predict(features)
        
        pred_x = 0
        pred_y = 0
        if prediction is not None:
            pred_x, pred_y = prediction
        
            # Draw gaze point
            gaze_x, gaze_y = int(pred_x), int(pred_y)
            cv2.circle(tracking_display, (gaze_x, gaze_y), 20, (0, 255, 0), -1)
            cv2.circle(tracking_display, (gaze_x, gaze_y), 25, (255, 255, 255), 2)

            # Check bounds and update warnings
            if not (0 <= pred_x <= screen_w and 0 <= pred_y <= screen_h):
                if warning_start is None:
                    warning_start = time.time()
                elif time.time() - warning_start > 1.5:
                    cv2.putText(tracking_display, "⚠ OFF-SCREEN!", 
                              (screen_w//2-200, screen_h//2), 
                              cv2.FONT_HERSHEY_DUPLEX, 2, (0,0,255), 3)
                    log.append([time.strftime("%H:%M:%S"), pred_x, pred_y, "Off-screen"])
            else:
                warning_start = None
            
            # Check if recalibration is needed
            if tracker.needs_recalibration:
                cv2.putText(tracking_display, "Recalibrating...", 
                          (50, 150), cv2.FONT_HERSHEY_DUPLEX, 
                          1, (0, 255, 255), 2)
                
                # Store stable points for recalibration
                if len(tracker.position_history) >= 30:
                    recent_pos = np.array(tracker.position_history[-30:])
                    if np.std(recent_pos, axis=0).mean() < 30:  # Stable position
                        recalibration_points.append([features, (int(pred_x), int(pred_y))])
                        
                        if len(recalibration_points) >= 5:  # Enough points to recalibrate
                            tracker.calibrate(recalibration_points)
                            recalibration_points = []
                            print("Recalibration complete")
            
            # Add coordinates display
            font_scale = screen_h / 1080
            cv2.putText(tracking_display, f"X: {int(pred_x)}, Y: {int(pred_y)}", 
                       (50, 50), cv2.FONT_HERSHEY_DUPLEX, 
                       font_scale, (255, 255, 255), 2)
        


    cv2.imshow("Gaze Monitor", tracking_display)
    if cv2.waitKey(1) & 0xFF == 27: break

cap.release()
cv2.destroyAllWindows()

pd.DataFrame(log, columns=["time","x","y","event"]).to_csv("gaze_log.csv", index=False)
print("Logs saved to gaze_log.csv")
