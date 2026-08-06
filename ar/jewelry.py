import numpy as np


def crop_to_content(rgba_image, alpha_threshold=60):
    """Crop away transparent padding around the actual product so a
    reference image sent to ai_enhance.py's AI enhancement flow isolates
    just the product, not empty margin.

    alpha_threshold is deliberately not near-zero: rembg cutouts usually
    leave a soft, semi-transparent feathered edge a few pixels wide. A very
    low threshold (e.g. 10, ~4% opacity) counts almost that entire faint
    fade as "content", so the crop's edges land slightly outside where the
    product is actually visible. Requiring ~25% opacity trims that
    near-invisible fringe.
    """
    alpha = rgba_image[:, :, 3]
    ys, xs = np.where(alpha > alpha_threshold)
    if len(xs) == 0 or len(ys) == 0:
        return rgba_image
    x1, x2 = xs.min(), xs.max() + 1
    y1, y2 = ys.min(), ys.max() + 1
    return rgba_image[y1:y2, x1:x2]
