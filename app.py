import cv2
import mediapipe as mp
import numpy as np
from flask import Flask, render_template, Response
from flask_socketio import SocketIO

app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Open Camera
cap = cv2.VideoCapture(0)

# Track previous nose position
prev_nose_y = None
nodding_threshold = 0.005  # Sensitivity

def detect_head_movement():
    global prev_nose_y

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)

        movement = "Straight"
        movementx = "Straight"
        tilt_angle = 0
        nod_angle = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                # Draw face mesh points
                for landmark in face_landmarks.landmark:
                    x, y = int(landmark.x * frame.shape[1]), int(landmark.y * frame.shape[0])
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Get Key Points
                left_eye = np.array([face_landmarks.landmark[33].x, face_landmarks.landmark[33].y])
                right_eye = np.array([face_landmarks.landmark[263].x, face_landmarks.landmark[263].y])
                nose = np.array([face_landmarks.landmark[1].x, face_landmarks.landmark[1].y])

                # **Calculate horizontal tilt angle (Left/Right)**
                dx = right_eye[0] - left_eye[0]
                dy = right_eye[1] - left_eye[1]
                tilt_angle = np.arctan2(dy, dx) * 180 / np.pi  # Angle in degrees

                # **Determine left/right tilt**
                if tilt_angle > 10:
                    movement = "Tilted Right"
                elif tilt_angle < -10:
                    movement = "Tilted Left"

                # **Calculate vertical nod angle (Up/Down)**
                eye_center_y = (left_eye[1] + right_eye[1]) / 2  # Midpoint of both eyes
                nod_angle = (nose[1] - eye_center_y) * 100  # Convert to degrees scale

                # **Determine up/down nod**
                if prev_nose_y is not None:
                    diff_y = nose[1] - prev_nose_y
                    if diff_y > nodding_threshold:
                        movementx = "Nodding Down"
                    elif diff_y < -nodding_threshold:
                        movementx = "Nodding Up"

                prev_nose_y = nose[1]

                # **Display movement text properly**
                cv2.putText(frame, f"Movement: {movement}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                cv2.putText(frame, f"Nodding: {movementx}", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
                cv2.putText(frame, f"Tilt Angle: {round(tilt_angle,2)}°", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
                cv2.putText(frame, f"Nod Angle: {round(nod_angle,2)}°", (20, 140), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Send movement & angle data to frontend
        socketio.emit("head_movement", {
            "movement": movement, 
            "movementx": movementx, 
            "tilt_angle": round(tilt_angle, 2), 
            "nod_angle": round(nod_angle, 2)
        })

        # Encode frame for streaming
        _, buffer = cv2.imencode(".jpg", frame)
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/video_feed")
def video_feed():
    return Response(detect_head_movement(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    socketio.run(app, host="0.0.0.0", port=5000, debug=True)
