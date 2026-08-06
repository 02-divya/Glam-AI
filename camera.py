import threading
import cv2
import mediapipe as mp
from analysis.skin_tone import get_roi, dominant_color, LEFT_CHEEK_IDX, SkinToneSmoother, analyze_skin_tone
from ar.lipstick import apply_lipstick, hex_to_bgr
from ar.foundation import apply_foundation

smoother = SkinToneSmoother(window_size=15, switch_threshold=5)

# Shared lipstick color state — no default shade, only set via product upload
current_lipstick = {"hex": None}

# Shared foundation state — set via /upload_foundation route
current_foundation = {"enabled": False, "hex": None}

# Jewelry and clothing have no live/CV rendering path — Jewelry no longer
# renders live at all (see the Home/Makeup/Jewelry/Clothes restructure),
# and clothing never did. Each is just a reference image handed straight
# to ai_enhance.py's capture-only AI enhancement flow.
uploaded_earring = {"rgba": None}    # set via /upload_earring route in app.py
uploaded_necklace = {"rgba": None}   # set via /upload_necklace route in app.py
uploaded_clothing = {"rgba": None}   # set via /upload_clothing route in app.py


def set_uploaded_earring(rgba_image):
    """Called from app.py after background removal to set the active earring reference image."""
    uploaded_earring["rgba"] = rgba_image


def set_uploaded_necklace(rgba_image):
    """Called from app.py after background removal to set the active necklace reference image."""
    uploaded_necklace["rgba"] = rgba_image


def set_uploaded_clothing(rgba_image):
    """Called from app.py after background removal to set the active clothing reference image."""
    uploaded_clothing["rgba"] = rgba_image


mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Separate MediaPipe instance for the static-photo path (render_static_photo,
# below), never touched by the live loop. Flask runs threaded=True, and a
# single MediaPipe solution instance isn't safe to call concurrently from
# two threads — the live loop is continuously calling .process() on
# `face_mesh` above, so a static-photo request landing on a different
# thread must not share it.
photo_face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)
photo_render_lock = threading.Lock()

# Shared state so app.py / frontend can read the latest results
latest_status = {"faces_detected": 0}
latest_skin_tone = {"hex": None, "undertone": None}

# Cache of the most recently rendered frame's JPEG bytes — this is what
# powers server-side photo capture. Capturing on the server (rather than
# trying to grab a frame from the browser's <img> stream via canvas) avoids
# a real browser limitation: some browsers only let canvas.drawImage() see
# the FIRST frame ever received from a multipart/x-mixed-replace stream,
# not the current one — even though the displayed image visually updates.
latest_frame_jpeg = {"bytes": None}

# Most recent RAW (pre-overlay) live frame — what "Use Live Capture" and
# the "Capture & Enhance with AI" flow both grab (via /use_camera_capture
# and /enhance_photo respectively), so they capture the person, not
# whatever lipstick/foundation happens to be active at that moment on the
# live feed.
latest_raw_frame = {"bgr": None}

# The working static photo, shared across ALL THREE category pages (set via
# /upload_user_photo or /use_camera_capture, cleared via /clear_photo) —
# picking a photo on one page carries over to the others, since it's one
# global "which photo am I working with" choice, not a per-page setting.
# When set, it takes priority over the live feed for: Makeup's rendered
# preview (render_static_photo), skin-tone/status reporting (see app.py's
# mode-aware /status and /skin_tone), and the AI enhance flow.
active_photo = {"bgr": None}

# One-shot face-detection/skin-tone result for active_photo, updated each
# time render_static_photo() runs — no smoothing needed since this is a
# single deterministic render, not a stream.
photo_status = {"faces_detected": 0, "hex": None, "undertone": None}

ANALYZE_EVERY_N_FRAMES = 10  # skin tone is expensive; don't run it every frame


def apply_face_overlays(frame, landmarks, w, h):
    """Applies foundation and lipstick for one detected face, in the fixed
    layer order that keeps lipstick visible on top of the foundation
    base."""
    if current_foundation["enabled"] and current_foundation["hex"]:
        frame = apply_foundation(frame, landmarks, w, h,
                                  color_bgr=hex_to_bgr(current_foundation["hex"]),
                                  alpha=0.35)

    if current_lipstick["hex"]:
        lip_color_bgr = hex_to_bgr(current_lipstick["hex"])
        frame = apply_lipstick(frame, landmarks, w, h,
                                color_bgr=lip_color_bgr, alpha=0.7)
    return frame


def render_static_photo():
    """Runs the makeup overlay pipeline once against active_photo and
    returns the rendered JPEG bytes, or None if there's no active photo.
    Also updates `photo_status` with a one-shot face/skin-tone result for
    that photo. Shared by all three category pages — whichever one is
    currently showing the mirror just fetches this same render."""
    if active_photo["bgr"] is None:
        return None

    with photo_render_lock:
        frame = active_photo["bgr"].copy()
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        face_results = photo_face_mesh.process(rgb)

        if face_results.multi_face_landmarks:
            landmarks = face_results.multi_face_landmarks[0].landmark
            frame = apply_face_overlays(frame, landmarks, w, h)

            photo_status["faces_detected"] = len(face_results.multi_face_landmarks)
            skin_result = analyze_skin_tone(frame, landmarks, w, h)
            if skin_result:
                photo_status["hex"] = skin_result["hex"]
                photo_status["undertone"] = skin_result["undertone"]
        else:
            photo_status["faces_detected"] = 0

        ok, buffer = cv2.imencode('.jpg', frame)
        if not ok:
            return None
        return buffer.tobytes()


def current_skin_tone():
    """Whichever skin-tone reading is authoritative right now: the static
    active_photo's one-shot analysis if a photo is active, otherwise the
    continuously-updated live reading. Used everywhere matching happens
    (lipstick/foundation/clothing match, best-deal-finder) so they stay
    consistent with whatever the mirror is currently showing, rather than
    always defaulting to the live feed even when the user deliberately
    switched to a specific photo."""
    if active_photo["bgr"] is not None:
        return photo_status.get("hex"), photo_status.get("undertone")
    return latest_skin_tone.get("hex"), latest_skin_tone.get("undertone")


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
        latest_raw_frame["bgr"] = frame.copy()  # pre-overlay, for Capture & Enhance
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:
            latest_status["faces_detected"] = len(results.multi_face_landmarks)

            for landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                frame = apply_face_overlays(frame, landmarks.landmark, w, h)

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
        latest_frame_jpeg["bytes"] = frame_bytes
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

    cap.release()
