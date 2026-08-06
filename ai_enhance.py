"""
Sends a captured/uploaded photo, plus reference images of any jewelry or
clothing the user uploaded, to OpenAI's image edit endpoint (GPT image
models) to produce a single photorealistic "wearing it" render.

This is fundamentally different from the deterministic CV overlay
pipeline in ar/*.py: instead of geometrically compositing exact pixels at
a calibrated position, a diffusion model reinterprets the reference
image's product onto the person's photo, with real lighting/shadow
interaction — at the cost of not guaranteeing exact product fidelity,
taking several seconds per call, and costing money per call. That's why
this is wired to a capture-only action (see app.py's /enhance_photo,
which only ever operates on the static active_photo, never the live
video stream).

API shape verified directly against the openai-python SDK source
(resources/images.py) rather than assumed from memory: images.edit's
`image` parameter accepts a list of up to 16 reference images for GPT
image models, and GPT image models always return base64-encoded results
(response.data[0].b64_json) — response_format/url mode isn't available
for these models.
"""

import base64
import os

import cv2
import numpy as np
from openai import OpenAI

MODEL = "gpt-image-1"
SIZE = "1024x1024"

BASE_INSTRUCTIONS = (
    "You are editing a real photo of a person. Keep their face, hair, "
    "skin tone, body proportions, pose, and the background EXACTLY as "
    "they appear in the original photo — do not regenerate, restyle, or "
    "alter identity. Only apply the specific change(s) described below, "
    "using the additional reference image(s) provided, and match the "
    "original photo's lighting, shadows, and perspective so the result "
    "reads as a single, consistent, unedited-looking photograph. Do not "
    "change the person's face, facial features, skin tone, expression, "
    "or identity in any way. The face must remain pixel-identical to the "
    "original photo. Only add the specified jewelry/clothing item — "
    "nothing else about the photo should change."
)

EARRING_INSTRUCTION = (
    "Add the earrings shown in one of the reference images to the "
    "person's ears, sized and angled naturally for their head."
)
NECKLACE_INSTRUCTION = (
    "Add the necklace shown in one of the reference images around the "
    "person's neck, resting naturally against their skin or clothing."
)
CLOTHING_INSTRUCTION = (
    "Replace the person's entire visible outfit with the garment shown in "
    "the reference image. Completely remove and replace their original "
    "clothing — none of their original garment should remain visible, "
    "including at the neckline, collar, or under any open areas of the "
    "new garment. If the reference garment (such as an open blazer) would "
    "naturally reveal an inner layer, add a simple, neutral, "
    "appropriately-colored top or blouse underneath that looks natural "
    "with the outfit — do not leave the original patterned/textured "
    "clothing showing through. Fit the new garment naturally to the "
    "person's body shape, pose, and proportions, matching the lighting "
    "and shadows of the original photo. Preserve the person's face, hair, "
    "skin, and the background exactly as they are — do not alter these."
)


def _encode_png(image, filename):
    """image is a BGR (3-channel, opaque photo) or BGRA (4-channel,
    rembg-cutout reference) numpy array — cv2 handles both transparently."""
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"Failed to encode {filename}")
    return (filename, buf.tobytes(), "image/png")


def enhance_captured_photo(photo_bgr, earring_rgba=None, necklace_rgba=None, clothing_rgba=None):
    """
    Send the captured/uploaded photo (BGR numpy array) plus whichever
    reference product images were provided to OpenAI's image edit
    endpoint, and return the resulting rendered image as a BGR numpy
    array.

    Returns (result_bgr, None) on success, or (None, error_message) on
    any failure. Never raises — callers turn a failure into a clean JSON
    error instead of a 500.
    """
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        return None, ("No OpenAI API key configured. Set the OPENAI_API_KEY "
                       "environment variable and restart the server to use AI enhancement.")

    if earring_rgba is None and necklace_rgba is None and clothing_rgba is None:
        return None, "Upload at least one item (earring, necklace, or clothing) before enhancing."

    instructions = [BASE_INSTRUCTIONS]
    images = [_encode_png(photo_bgr, "photo.png")]

    if earring_rgba is not None:
        instructions.append(EARRING_INSTRUCTION)
        images.append(_encode_png(earring_rgba, "earring_reference.png"))
    if necklace_rgba is not None:
        instructions.append(NECKLACE_INSTRUCTION)
        images.append(_encode_png(necklace_rgba, "necklace_reference.png"))
    if clothing_rgba is not None:
        instructions.append(CLOTHING_INSTRUCTION)
        images.append(_encode_png(clothing_rgba, "clothing_reference.png"))

    prompt = " ".join(instructions)

    try:
        client = OpenAI(api_key=api_key)
        response = client.images.edit(
            model=MODEL,
            image=images,
            prompt=prompt,
            size=SIZE,
            input_fidelity="high",  # spend more effort matching the person's actual face/features
        )
    except Exception as e:
        return None, f"AI enhancement failed: {e}"

    if not response.data or not response.data[0].b64_json:
        return None, "AI enhancement returned no image."

    raw = base64.b64decode(response.data[0].b64_json)
    arr = np.frombuffer(raw, np.uint8)
    result_bgr = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if result_bgr is None:
        return None, "Received an unreadable image from the AI service."

    return result_bgr, None
