import cv2
import mediapipe as mp
from analysis.skin_tone import get_roi, dominant_color, LEFT_CHEEK_IDX, SkinToneSmoother
from ar.lipstick import apply_lipstick, hex_to_bgr
from ar.jewelry import apply_jewelry, apply_uploaded_earrings
from ar.necklace import apply_necklace

smoother = SkinToneSmoother(window_size=15, switch_threshold=5)

# Shared lipstick color state — default classic red
current_lipstick = {"hex": "#c2185b"}

# Shared jewelry state. style: "stud", "hoop", or "uploaded". scale/horizontal/vertical tunable live.
current_jewelry = {"enabled": True, "style": "stud", "scale_adjust": 1.0, "horizontal_adjust": 0.0, "vertical_adjust": 0.0}
uploaded_earring = {"rgba": None}  # set via /upload_earring route in app.py

# Shared necklace state — offset_adjust/scale_adjust are tuned live via UI sliders
current_necklace = {"enabled": False, "offset_adjust": 0.0, "scale_adjust": 1.0}
uploaded_necklace = {"rgba": None}  # set via /upload_necklace route in app.py


def set_uploaded_earring(rgba_image):
    """Called from app.py after background removal to set the active earring image."""
    uploaded_earring["rgba"] = rgba_image
    current_jewelry["style"] = "uploaded"


def set_uploaded_necklace(rgba_image):
    """Called from app.py after background removal to set the active necklace image."""
    uploaded_necklace["rgba"] = rgba_image
    current_necklace["enabled"] = True

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_pose = mp.solutions.pose
pose = mp_pose.Pose(
    model_complexity=0,  # lightest model — we only need shoulder landmarks, not full-body detail
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Shared state so app.py / frontend can read the latest results
latest_status = {"faces_detected": 0}
latest_skin_tone = {"hex": None, "undertone": None}

ANALYZE_EVERY_N_FRAMES = 10  # skin tone is expensive; don't run it every frame


def generate_frames():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("ERROR: Could not open webcam. Check that it's not in use by "
              "another app, and that your OS granted camera permission.")
        return

    frame_count = 0

    while True:
        success, frame = cap.read()
        if not success:
            print("WARNING: Failed to read frame from webcam.")
            break

        frame = cv2.flip(frame, 1)  # mirror view feels natural
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        # Necklace is anchored to shoulders, not face — apply it first so
        # it sits "underneath" the face/lipstick/jewelry layer if they overlap
        if current_necklace["enabled"] and uploaded_necklace["rgba"] is not None:
            pose_results = pose.process(rgb)
            if pose_results.pose_landmarks:
                h, w, _ = frame.shape
                frame = apply_necklace(frame, pose_results.pose_landmarks.landmark, w, h,
                                        uploaded_necklace["rgba"],
                                        offset_adjust=current_necklace["offset_adjust"],
                                        scale_adjust=current_necklace["scale_adjust"])

        if results.multi_face_landmarks:
            latest_status["faces_detected"] = len(results.multi_face_landmarks)

            for landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                # Apply lipstick BEFORE drawing mesh dots so dots stay visible on top
                lip_color_bgr = hex_to_bgr(current_lipstick["hex"])
                frame = apply_lipstick(frame, landmarks.landmark, w, h,
                                        color_bgr=lip_color_bgr, alpha=0.45)

                # Apply jewelry — either a real uploaded photo or a drawn placeholder
                if current_jewelry["enabled"]:
                    if current_jewelry["style"] == "uploaded" and uploaded_earring["rgba"] is not None:
                        frame = apply_uploaded_earrings(frame, landmarks.landmark, w, h,
                                                         uploaded_earring["rgba"],
                                                         scale_adjust=current_jewelry["scale_adjust"],
                                                         horizontal_adjust=current_jewelry["horizontal_adjust"],
                                                         vertical_adjust=current_jewelry["vertical_adjust"])
                    else:
                        frame = apply_jewelry(frame, landmarks.landmark, w, h,
                                               style=current_jewelry["style"],
                                               horizontal_adjust=current_jewelry["horizontal_adjust"],
                                               vertical_adjust=current_jewelry["vertical_adjust"])

                # Draw the mesh dots
                for lm in landmarks.landmark:
                    x, y = int(lm.x * w), int(lm.y * h)
                    cv2.circle(frame, (x, y), 1, (0, 255, 0), -1)

                # Run skin tone analysis periodically (not every frame — it's
                # heavier work and doesn't need to be instant)
                frame_count += 1
                if frame_count % ANALYZE_EVERY_N_FRAMES == 0:
                    roi = get_roi(frame, landmarks.landmark, LEFT_CHEEK_IDX, w, h)
                    raw_result = dominant_color(roi)
                    if raw_result:
                        _, rgb = raw_result
                        smoothed = smoother.update(rgb)
                        latest_skin_tone["hex"] = smoothed["hex"]
                        latest_skin_tone["undertone"] = smoothed["undertone"]
        else:
            latest_status["faces_detected"] = 0

        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            continue
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()