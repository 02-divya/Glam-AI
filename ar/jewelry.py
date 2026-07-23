import cv2
import numpy as np

# MediaPipe FaceMesh landmark indices near the cheek/ear-attachment area
LEFT_EAR_IDX = 234
RIGHT_EAR_IDX = 454

# Landmarks used to measure face height, so the earlobe offset scales
# correctly whether the person is close to or far from the camera
FOREHEAD_IDX = 10
CHIN_IDX = 152
NOSE_TIP_IDX = 1  # used as a face-center reference for horizontal offset

# Face mesh has no true earlobe landmark, so we nudge the nearest available
# point downward. This fraction of total face height approximates the
# distance from cheekbone height down to where an earring actually hangs.
EARLOBE_OFFSET_RATIO = 0.22


def _get_earlobe_point(landmarks, idx, w, h, horizontal_adjust=0.0, vertical_adjust=0.0):
    """Return (x, y) for an earring anchor, offset down from the raw
    landmark toward the earlobe, scaled by the person's face size.
    horizontal_adjust (can be negative) pushes the point outward/inward
    from the face center. vertical_adjust (can be negative) nudges the
    point up/down further, on top of the base earlobe offset."""
    lm = landmarks[idx]
    x, y = lm.x * w, lm.y * h

    forehead = landmarks[FOREHEAD_IDX]
    chin = landmarks[CHIN_IDX]
    face_height = abs((chin.y - forehead.y) * h)

    y += face_height * (EARLOBE_OFFSET_RATIO + vertical_adjust)

    nose = landmarks[NOSE_TIP_IDX]
    center_x = nose.x * w
    direction = 1 if x > center_x else -1
    x += direction * face_height * horizontal_adjust

    return int(x), int(y)


def draw_stud_earring(frame, x, y, size=10, color_bgr=(0, 215, 255)):
    """Draw a simple gold stud earring — a filled circle with a highlight."""
    cv2.circle(frame, (x, y), size, color_bgr, -1, lineType=cv2.LINE_AA)
    highlight = (min(color_bgr[0] + 60, 255), min(color_bgr[1] + 40, 255), min(color_bgr[2] + 40, 255))
    cv2.circle(frame, (x - size // 3, y - size // 3), max(size // 4, 1), highlight, -1, lineType=cv2.LINE_AA)


def draw_hoop_earring(frame, x, y, radius=14, thickness=3, color_bgr=(0, 215, 255)):
    """Draw a simple gold hoop earring hanging below the anchor point."""
    center = (x, y + radius)
    cv2.circle(frame, center, radius, color_bgr, thickness, lineType=cv2.LINE_AA)


def apply_jewelry(frame, landmarks, w, h, style="stud", color_bgr=(0, 215, 255), horizontal_adjust=0.0, vertical_adjust=0.0):
    """Draw synthetic earrings (stud/hoop) at both earlobe-approximated positions."""
    for idx in (LEFT_EAR_IDX, RIGHT_EAR_IDX):
        x, y = _get_earlobe_point(landmarks, idx, w, h, horizontal_adjust=horizontal_adjust, vertical_adjust=vertical_adjust)
        if style == "hoop":
            draw_hoop_earring(frame, x, y, color_bgr=color_bgr)
        else:
            draw_stud_earring(frame, x, y, color_bgr=color_bgr)
    return frame


def overlay_image_at_point(frame, x, y, overlay_rgba, target_width_px=40):
    """
    Alpha-blend a BGRA image onto frame, centered horizontally at (x,y)
    and hanging downward from that point (matches how an earring hangs
    from the earlobe).
    """
    if overlay_rgba is None:
        return frame

    oh, ow = overlay_rgba.shape[:2]
    scale = target_width_px / ow
    new_w, new_h = max(int(ow * scale), 1), max(int(oh * scale), 1)
    resized = cv2.resize(overlay_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    x1, y1 = x - new_w // 2, y
    x2, y2 = x1 + new_w, y1 + new_h

    # Clip to frame bounds instead of skipping entirely, so partial
    # overlaps near frame edges still render what's visible
    fx1, fy1 = max(x1, 0), max(y1, 0)
    fx2, fy2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
    if fx1 >= fx2 or fy1 >= fy2:
        return frame  # fully out of bounds

    ox1, oy1 = fx1 - x1, fy1 - y1
    ox2, oy2 = ox1 + (fx2 - fx1), oy1 + (fy2 - fy1)
    region = resized[oy1:oy2, ox1:ox2]

    if region.shape[2] == 4:
        alpha = region[:, :, 3:4] / 255.0
    else:
        alpha = np.ones((region.shape[0], region.shape[1], 1))

    frame_region = frame[fy1:fy2, fx1:fx2].astype(float)
    blended = alpha * region[:, :, :3].astype(float) + (1 - alpha) * frame_region
    frame[fy1:fy2, fx1:fx2] = blended.astype(np.uint8)
    return frame


def apply_uploaded_earrings(frame, landmarks, w, h, overlay_rgba, scale_adjust=1.0, horizontal_adjust=0.0, vertical_adjust=0.0):
    """Overlay a real uploaded earring image at both earlobe-approximated positions.
    Size is proportional to face width so it scales sensibly regardless of
    how close/zoomed the camera is, with scale_adjust as a live user-tunable multiplier."""
    left = landmarks[LEFT_EAR_IDX]
    right = landmarks[RIGHT_EAR_IDX]
    face_width_px = abs((right.x - left.x) * w)
    target_width_px = max(int(face_width_px * 0.22 * scale_adjust), 10)

    for idx in (LEFT_EAR_IDX, RIGHT_EAR_IDX):
        x, y = _get_earlobe_point(landmarks, idx, w, h, horizontal_adjust=horizontal_adjust, vertical_adjust=vertical_adjust)
        frame = overlay_image_at_point(frame, x, y, overlay_rgba, target_width_px)
    return frame