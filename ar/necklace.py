import numpy as np
from ar.jewelry import overlay_image_at_point

# MediaPipe Pose landmark indices for shoulders
LEFT_SHOULDER_IDX = 11
RIGHT_SHOULDER_IDX = 12

# Base values — actual position/size also adjusted live via UI sliders,
# since necklace photos vary a lot in how much chain sits above the pendant
NECK_OFFSET_RATIO = 0.05
NECKLACE_WIDTH_RATIO = 0.55


def get_neck_anchor(pose_landmarks, w, h, offset_adjust=0.0):
    """
    Returns (x, y, shoulder_width_px) using the midpoint between shoulders,
    offset down toward where a necklace sits. offset_adjust is an extra
    ratio (can be negative) added on top of the base offset, driven by
    a live slider so it can be tuned per necklace image.
    Returns None if shoulders aren't confidently detected.
    """
    left = pose_landmarks[LEFT_SHOULDER_IDX]
    right = pose_landmarks[RIGHT_SHOULDER_IDX]

    if getattr(left, "visibility", 1.0) < 0.5 or getattr(right, "visibility", 1.0) < 0.5:
        return None

    lx, ly = left.x * w, left.y * h
    rx, ry = right.x * w, right.y * h

    shoulder_width = abs(rx - lx)
    mid_x = (lx + rx) / 2
    mid_y = (ly + ry) / 2 + shoulder_width * (NECK_OFFSET_RATIO + offset_adjust)

    return int(mid_x), int(mid_y), shoulder_width


def apply_necklace(frame, pose_landmarks, w, h, overlay_rgba, offset_adjust=0.0, scale_adjust=1.0):
    """Overlay a real uploaded necklace image anchored at the neck/collarbone."""
    if overlay_rgba is None or pose_landmarks is None:
        return frame

    anchor = get_neck_anchor(pose_landmarks, w, h, offset_adjust=offset_adjust)
    if anchor is None:
        return frame

    x, y, shoulder_width = anchor
    target_width_px = max(int(shoulder_width * NECKLACE_WIDTH_RATIO * scale_adjust), 10)

    return overlay_image_at_point(frame, x, y, overlay_rgba, target_width_px=target_width_px)