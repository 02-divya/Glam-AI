import cv2
import numpy as np

# Accurate outer-lip contour landmark indices from MediaPipe FaceMesh
# (this is the actual lip boundary, not just a raw numeric range)
LIPS_OUTER_IDX = [
    61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291,
    375, 321, 405, 314, 17, 84, 181, 91, 146
]


def apply_lipstick(frame, landmarks, w, h, color_bgr=(0, 0, 200), alpha=0.45):
    """
    Overlay a translucent color on the lips using a convex-hull mask
    built from the outer lip landmarks, alpha-blended onto the frame.
    """
    points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in LIPS_OUTER_IDX
    ])

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)

    # Slight blur on the mask edge so the tint doesn't look hard-edged/pasted-on
    mask = cv2.GaussianBlur(mask, (5, 5), 0)

    color_layer = np.zeros_like(frame)
    color_layer[:] = color_bgr

    # Blend only within the mask region
    mask_f = (mask.astype(float) / 255.0)[..., None]
    blended = frame.astype(float) * (1 - alpha * mask_f) + color_layer.astype(float) * (alpha * mask_f)
    frame[:] = blended.astype(np.uint8)
    return frame


def hex_to_bgr(hex_code: str):
    """Convert '#rrggbb' to an (B, G, R) tuple for OpenCV."""
    hex_code = hex_code.lstrip('#')
    r = int(hex_code[0:2], 16)
    g = int(hex_code[2:4], 16)
    b = int(hex_code[4:6], 16)
    return (b, g, r)