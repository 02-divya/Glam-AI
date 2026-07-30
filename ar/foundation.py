import cv2
import numpy as np

# MediaPipe FaceMesh face-oval outline (the standard loop used for a
# full-face mask). Deliberately does NOT exclude eyes/lips — a real
# foundation application would skip those, but that needs finer per-feature
# masking. Noted as a known simplification for this MVP.
FACE_OVAL_IDX = [
    10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365,
    379, 378, 400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93,
    234, 127, 162, 21, 54, 103, 67, 109
]


def apply_foundation(frame, landmarks, w, h, color_bgr, alpha=0.55):
    """
    Shift the face's color toward the target foundation shade while
    preserving the face's natural lighting (highlights/shadows).

    A flat alpha-blend (mixing in a solid color at fixed opacity) ignores
    lighting entirely, which produces a blotchy, muddy "painted on" look —
    bright and shadowed areas of the face get muddied unevenly. Instead,
    this works in Lab color space, which separates lightness (L) from
    color (a/b): we shift color strongly toward the target shade, but only
    lightly touch lightness, so the face's real shading still comes
    through and it reads as skin rather than a flat mask.
    """
    points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h))
        for i in FACE_OVAL_IDX
    ])

    mask = np.zeros(frame.shape[:2], dtype=np.uint8)
    hull = cv2.convexHull(points)
    cv2.fillConvexPoly(mask, hull, 255)
    mask = cv2.GaussianBlur(mask, (25, 25), 0)  # soft edge, avoids a hard cutout look

    mask_f = (mask.astype(float) / 255.0)

    lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(float)

    target_patch = np.uint8([[color_bgr]])
    target_lab = cv2.cvtColor(target_patch, cv2.COLOR_BGR2LAB)[0][0].astype(float)

    # Color (a/b channels) shifts strongly toward the target shade.
    # Lightness (L channel) barely shifts — this is what keeps the face's
    # real highlights/shadows intact instead of flattening them.
    chroma_alpha = alpha
    lightness_alpha = alpha * 0.2

    new_lab = lab_frame.copy()
    new_lab[..., 0] = lab_frame[..., 0] * (1 - lightness_alpha * mask_f) + target_lab[0] * (lightness_alpha * mask_f)
    new_lab[..., 1] = lab_frame[..., 1] * (1 - chroma_alpha * mask_f) + target_lab[1] * (chroma_alpha * mask_f)
    new_lab[..., 2] = lab_frame[..., 2] * (1 - chroma_alpha * mask_f) + target_lab[2] * (chroma_alpha * mask_f)

    new_lab = np.clip(new_lab, 0, 255).astype(np.uint8)
    result = cv2.cvtColor(new_lab, cv2.COLOR_LAB2BGR)
    frame[:] = result
    return frame