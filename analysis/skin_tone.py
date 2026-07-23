import cv2
import numpy as np
from sklearn.cluster import KMeans

# MediaPipe FaceMesh landmark indices around the left cheek
LEFT_CHEEK_IDX = [50, 101, 118, 117]


def get_roi(frame, landmarks, indices, w, h):
    """Extract a bounding-box region of interest from given landmark indices."""
    points = np.array([
        (int(landmarks[i].x * w), int(landmarks[i].y * h)) for i in indices
    ])
    x, y, rw, rh = cv2.boundingRect(points)
    # Guard against zero-size or out-of-frame boxes
    x, y = max(x, 0), max(y, 0)
    rw, rh = max(rw, 1), max(rh, 1)
    roi = frame[y:y + rh, x:x + rw]
    return roi


def dominant_color(roi, k=3):
    """Return (hex_code, (r,g,b)) of the dominant cluster in the ROI, ignoring
    very dark/very bright pixels (shadow/highlight noise)."""
    if roi is None or roi.size == 0:
        return None

    pixels = roi.reshape(-1, 3).astype(float)
    brightness = pixels.sum(axis=1)
    mask = (brightness > 60) & (brightness < 720)
    filtered = pixels[mask] if mask.any() else pixels

    if len(filtered) < k:
        return None

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(filtered)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[counts.argmax()]
    b, g, r = dominant
    hex_code = '#%02x%02x%02x' % (int(r), int(g), int(b))
    return hex_code, (r, g, b)


def classify_undertone(rgb):
    """Simple heuristic undertone classifier based on red/blue balance."""
    r, g, b = rgb
    if r > b and (r - b) > 15:
        return "Warm"
    elif b > r:
        return "Cool"
    else:
        return "Neutral"


def analyze_skin_tone(frame, landmarks, w, h):
    """Full pipeline: ROI -> dominant color -> undertone classification.
    Returns a dict, or None if analysis wasn't possible this frame."""
    roi = get_roi(frame, landmarks, LEFT_CHEEK_IDX, w, h)
    result = dominant_color(roi)
    if result is None:
        return None
    hex_code, rgb = result
    undertone = classify_undertone(rgb)
    return {"hex": hex_code, "undertone": undertone}


class SkinToneSmoother:
    """
    Averages recent skin-tone readings over a rolling window so the
    displayed color/undertone doesn't flicker between frames due to
    lighting noise or tiny head movement.

    Also requires the SAME undertone to repeat several times in a row
    before switching the displayed label — this stops "Warm" flickering
    to "Cool" and back on borderline readings.
    """

    def __init__(self, window_size=15, switch_threshold=5):
        self.window_size = window_size
        self.switch_threshold = switch_threshold
        self.rgb_history = []
        self.current_undertone = None
        self.pending_undertone = None
        self.pending_count = 0

    def update(self, rgb):
        # Rolling average of RGB values
        self.rgb_history.append(rgb)
        if len(self.rgb_history) > self.window_size:
            self.rgb_history.pop(0)

        avg = np.mean(self.rgb_history, axis=0)
        avg_r, avg_g, avg_b = avg
        hex_code = '#%02x%02x%02x' % (int(avg_r), int(avg_g), int(avg_b))
        raw_undertone = classify_undertone((avg_r, avg_g, avg_b))

        # Require the new undertone to persist before switching displayed label
        if raw_undertone == self.current_undertone:
            self.pending_undertone = None
            self.pending_count = 0
        elif raw_undertone == self.pending_undertone:
            self.pending_count += 1
            if self.pending_count >= self.switch_threshold:
                self.current_undertone = raw_undertone
                self.pending_undertone = None
                self.pending_count = 0
        else:
            self.pending_undertone = raw_undertone
            self.pending_count = 1

        if self.current_undertone is None:
            self.current_undertone = raw_undertone

        return {"hex": hex_code, "undertone": self.current_undertone}