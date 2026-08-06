"""
Extracts the dominant color from an uploaded product photo (lipstick,
foundation, or clothing) and judges whether it's a good match for the
person's detected skin tone.

MATCHING METHODOLOGY (this matters — each product type is judged
differently, matching how real style/makeup matching actually works):

- FOUNDATION should closely match the person's actual skin color. We measure
  this with perceptual color distance (CIE Lab space, not raw RGB, since Lab
  distance correlates much better with how different two colors actually
  look to a human eye).

- LIPSTICK is not supposed to match skin color — it's a deliberate contrast.
  What actually matters is undertone compatibility (a shade with warm
  undertones generally suits warm-undertone skin better, etc). So lipstick
  matching compares undertone classification, not raw color distance.

- CLOTHING, like lipstick, isn't judged by color-closeness to skin either —
  it's judged by undertone compatibility (a common styling heuristic: warm
  colors are generally said to flatter warm-undertone skin more, and
  likewise for cool). The undertone tag on a garment's color is OUR OWN
  color-family classification (via classify_undertone), the same heuristic
  used everywhere else in this file — not an official styling authority's
  judgement.
"""

import cv2
import numpy as np
from sklearn.cluster import KMeans


def extract_product_color(image_bytes: bytes):
    """
    Given raw uploaded image bytes, remove the background (isolating the
    product from whatever surface/hand/packaging is around it) and return
    the dominant color as (hex_code, (r, g, b)). Returns None on failure.
    """
    try:
        from rembg import remove
        output_bytes = remove(image_bytes)
    except Exception as e:
        raise RuntimeError(f"Background removal failed: {e}")

    arr = np.frombuffer(output_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None or img.shape[2] != 4:
        return None

    alpha = img[:, :, 3]
    mask = alpha > 128
    if mask.sum() < 20:
        return None

    pixels = img[mask][:, :3].astype(float)  # BGR

    # Ignore near-black/near-white pixels (shadows, packaging highlights,
    # stray background fringing rembg didn't fully clear)
    brightness = pixels.sum(axis=1)
    filtered_mask = (brightness > 45) & (brightness < 720)
    filtered = pixels[filtered_mask] if filtered_mask.sum() >= 10 else pixels

    k = min(3, len(filtered))
    if k < 1:
        return None

    kmeans = KMeans(n_clusters=k, n_init=10, random_state=42).fit(filtered)
    counts = np.bincount(kmeans.labels_)
    dominant = kmeans.cluster_centers_[counts.argmax()]
    b, g, r = dominant
    hex_code = '#%02x%02x%02x' % (int(r), int(g), int(b))
    return hex_code, (int(r), int(g), int(b))


def _rgb_to_lab(rgb):
    """Convert a single (r,g,b) tuple to CIE Lab via OpenCV."""
    r, g, b = rgb
    patch = np.uint8([[[b, g, r]]])  # OpenCV expects BGR
    lab = cv2.cvtColor(patch, cv2.COLOR_BGR2LAB)[0][0]
    return lab.astype(float)


def color_distance(rgb1, rgb2):
    """Perceptual distance between two RGB colors using CIE Lab space."""
    lab1 = _rgb_to_lab(rgb1)
    lab2 = _rgb_to_lab(rgb2)
    return float(np.linalg.norm(lab1 - lab2))


def classify_undertone(rgb):
    """Same heuristic used for skin tone: red/blue balance."""
    r, g, b = rgb
    if r > b and (r - b) > 15:
        return "Warm"
    elif b > r:
        return "Cool"
    else:
        return "Neutral"


def judge_foundation_match(product_rgb, skin_rgb):
    """Foundation should closely match actual skin color."""
    distance = color_distance(product_rgb, skin_rgb)
    if distance < 12:
        verdict, quality = "Excellent match — very close to your skin tone.", "excellent"
    elif distance < 22:
        verdict, quality = "Good match — close to your skin tone.", "good"
    elif distance < 35:
        verdict, quality = "Fair match — noticeably different from your skin tone, may look slightly off.", "fair"
    else:
        verdict, quality = "Not a great match — this shade looks quite different from your skin tone.", "poor"
    return {"verdict": verdict, "quality": quality, "distance": round(distance, 1)}


def _get_saturation(rgb):
    """Returns HSV saturation (0-255) for an RGB color."""
    r, g, b = rgb
    patch = np.uint8([[[b, g, r]]])
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)[0][0]
    return float(hsv[1])


def judge_lipstick_match(product_rgb, skin_undertone, skin_rgb=None):
    """
    Lipstick is primarily judged by undertone compatibility, not color
    closeness to skin (lipstick is meant to contrast with skin, not match
    it). BUT undertone alone doesn't catch everything a real match should:
    a shade can share the "correct" undertone and still look washed out if
    it's too low-contrast against the skin (too close in lightness) or too
    desaturated/dusty. When skin_rgb is available, we check for that too
    and downgrade the verdict with an explicit warning — this is the
    difference between "technically the right undertone family" and
    "will actually look good on camera."
    """
    product_undertone = classify_undertone(product_rgb)

    if product_undertone == skin_undertone:
        verdict = f"This shade's {product_undertone.lower()} tones suit your {skin_undertone.lower()} undertone well."
        quality = "good"
    elif "Neutral" in (product_undertone, skin_undertone):
        verdict = f"This is a {product_undertone.lower()}-toned shade — neutral-friendly, should work reasonably well with your {skin_undertone.lower()} undertone."
        quality = "fair"
    else:
        verdict = (f"This is a {product_undertone.lower()}-toned shade, but you have a {skin_undertone.lower()} "
                   f"undertone — it may clash slightly. A {skin_undertone.lower()}-toned shade would likely suit you better.")
        quality = "poor"

    # Contrast/saturation check — catches "right undertone, still washes you out"
    if skin_rgb is not None and quality != "poor":
        lightness_diff = abs(_rgb_to_lab(product_rgb)[0] - _rgb_to_lab(skin_rgb)[0])
        saturation = _get_saturation(product_rgb)

        low_contrast = lightness_diff < 12
        low_saturation = saturation < 60

        if low_contrast and low_saturation:
            verdict += (" That said, this particular shade is quite close in lightness to your skin tone and "
                        "fairly muted — it may look washed out or barely visible on camera rather than making "
                        "your lips pop. A more saturated or higher-contrast shade in the same undertone family "
                        "would likely show up better.")
            quality = "fair" if quality == "good" else quality
        elif low_contrast:
            verdict += (" Note: this shade is close in lightness to your skin tone, so it may read as subtle/"
                        "low-contrast rather than a bold lip — that may or may not be what you're going for.")

    return {"verdict": verdict, "quality": quality, "product_undertone": product_undertone}


def judge_clothing_match(product_rgb, skin_undertone):
    """
    Clothing is judged the same way lipstick is: by undertone compatibility,
    not color-closeness to skin — see the module-level methodology note.
    Deliberately simpler than judge_lipstick_match (no contrast/saturation
    check): that check exists specifically to catch a lipstick "washing out"
    against skin at close range, which isn't a meaningful concern for an
    entire garment.
    """
    product_undertone = classify_undertone(product_rgb)

    if product_undertone == skin_undertone:
        verdict = f"This {product_undertone.lower()}-toned garment complements your {skin_undertone.lower()} undertone well."
        quality = "good"
    elif "Neutral" in (product_undertone, skin_undertone):
        verdict = (f"This is a {product_undertone.lower()}-toned garment — neutral-friendly, should work "
                   f"reasonably well with your {skin_undertone.lower()} undertone.")
        quality = "fair"
    else:
        verdict = (f"This is a {product_undertone.lower()}-toned garment, but you have a {skin_undertone.lower()} "
                   f"undertone — it may clash slightly. A {skin_undertone.lower()}-toned garment would likely suit you better.")
        quality = "poor"

    return {"verdict": verdict, "quality": quality, "product_undertone": product_undertone}