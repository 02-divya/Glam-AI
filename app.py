from dotenv import load_dotenv
load_dotenv()  # loads OPENAI_API_KEY (etc.) from a project-local .env file —
                # more reliable than shell/OS environment variables, which
                # don't propagate to processes already running when they're set

from flask import Flask, render_template, Response, jsonify, request
import base64
import numpy as np
import cv2
from camera import (generate_frames, latest_status,
                     current_lipstick, current_foundation,
                     latest_frame_jpeg, latest_raw_frame,
                     set_uploaded_earring, set_uploaded_necklace, set_uploaded_clothing,
                     set_uploaded_clothing_2,
                     uploaded_earring, uploaded_necklace, uploaded_clothing,
                     uploaded_clothing_2,
                     active_photo, photo_status as camera_photo_status,
                     render_static_photo, current_skin_tone)
from ar.jewelry import crop_to_content
from commerce.deal_finder import find_best_deal, find_closest_shade
from analysis.product_match import (extract_product_color, judge_foundation_match,
                                     judge_lipstick_match, judge_clothing_match)
from ai_enhance import enhance_captured_photo

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/snapshot')
def snapshot():
    """
    Returns the most recently rendered frame as a single static JPEG —
    this is what actually powers "Capture Photo" on the Makeup page.
    Capturing server-side like this (rather than the browser grabbing a
    frame from the live <img> stream) guarantees you get the exact
    current frame with all overlays applied, sidestepping a real browser
    quirk where canvas can't reliably read frames from an ongoing
    multipart image stream.
    """
    if latest_frame_jpeg["bytes"] is None:
        return jsonify({"error": "No frame available yet"}), 503
    return Response(latest_frame_jpeg["bytes"], mimetype='image/jpeg')


@app.route('/status')
def status():
    """Mode-aware: reports the static active_photo's one-shot face count
    if a photo is active, otherwise the continuously-updated live count —
    same reasoning as camera.current_skin_tone()."""
    if active_photo["bgr"] is not None:
        return jsonify({"faces_detected": camera_photo_status["faces_detected"]})
    return jsonify(latest_status)


@app.route('/skin_tone')
def skin_tone():
    hex_code, undertone = current_skin_tone()
    return jsonify({"hex": hex_code, "undertone": undertone})


@app.route('/upload_user_photo', methods=['POST'])
def upload_user_photo():
    """Sets the shared working photo from a user-uploaded picture of
    themselves. No background removal here — unlike product photos, this
    is the whole scene (person + surroundings). Shared across all three
    category pages — see camera.active_photo."""
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    raw_bytes = file.read()
    arr = np.frombuffer(raw_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        return jsonify({"success": False, "error": "Could not read image"}), 400

    active_photo["bgr"] = img
    return jsonify({"success": True})


@app.route('/use_camera_capture', methods=['POST'])
def use_camera_capture():
    """Sets the shared working photo from the current live webcam frame —
    the raw pre-overlay frame, not whatever's currently rendered on the
    live feed."""
    if latest_raw_frame["bgr"] is None:
        return jsonify({"success": False, "error": "No live camera frame available yet"}), 503

    active_photo["bgr"] = latest_raw_frame["bgr"].copy()
    return jsonify({"success": True})


@app.route('/clear_photo', methods=['POST'])
def clear_photo():
    active_photo["bgr"] = None
    return jsonify({"success": True})


@app.route('/photo_render')
def photo_render():
    jpeg_bytes = render_static_photo()
    if jpeg_bytes is None:
        return jsonify({"error": "No active photo"}), 404
    return Response(jpeg_bytes, mimetype='image/jpeg')


@app.route('/best_deal')
def best_deal():
    undertone = current_skin_tone()[1]
    product_type = request.args.get("product_type", "lipstick")

    if not undertone:
        return jsonify({"success": False, "error": "No skin tone detected yet — make sure your face is in frame."}), 400

    deal = find_best_deal(undertone, product_type)
    if not deal:
        return jsonify({"success": False, "error": f"No {product_type} matches found for undertone '{undertone}'."}), 404

    return jsonify({"success": True, "undertone": undertone, "deal": deal})


@app.route('/find_similar_shade', methods=['POST'])
def find_similar_shade():
    """Image-based alternative to /best_deal: upload a product photo instead
    of relying on detected undertone, extract its dominant color (same
    extract_product_color used by the lipstick/foundation match routes),
    and find the closest-colored catalog shade via find_closest_shade()."""
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    product_type = request.form.get('product_type', 'lipstick')
    raw_bytes = file.read()

    try:
        result = extract_product_color(raw_bytes)
    except RuntimeError as e:
        return jsonify({
            "success": False,
            "error": f"{e} First use requires internet access to download the rembg model."
        }), 500

    if result is None:
        return jsonify({"success": False, "error": "Couldn't isolate a clear product color from that photo. Try a closer, more clearly-lit shot."}), 400

    extracted_hex, rgb = result

    match = find_closest_shade(rgb, product_type)
    if match is None:
        return jsonify({"success": False, "error": f"No {product_type} shades in the catalog to compare against."}), 404

    return jsonify({
        "success": True,
        "extracted_hex": extracted_hex,
        "shade_name": match["shade_name"],
        "line_name": match["line_name"],
        "match_label": match["match_label"],
        "distance": match["distance"],
        "shade_hex": match["shade_hex"],
        "platform_links": match["platform_links"],
    })


@app.route('/set_lipstick', methods=['POST'])
def set_lipstick():
    data = request.get_json()
    hex_code = data.get('hex')
    if hex_code and len(hex_code) == 7 and hex_code.startswith('#'):
        current_lipstick['hex'] = hex_code
        return jsonify({"success": True, "hex": hex_code})
    return jsonify({"success": False, "error": "invalid hex"}), 400


@app.route('/upload_earring', methods=['POST'])
def upload_earring():
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    raw_bytes = file.read()

    try:
        # Lazy import — rembg pulls in onnxruntime, only load it if this
        # route is actually used
        from rembg import remove
        output_bytes = remove(raw_bytes)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Background removal failed: {e}. "
                     "First use requires internet access to download the rembg model."
        }), 500

    arr = np.frombuffer(output_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None or img.shape[2] != 4:
        return jsonify({"success": False, "error": "Could not process image"}), 400

    img = crop_to_content(img)
    set_uploaded_earring(img)
    return jsonify({"success": True})


@app.route('/upload_necklace', methods=['POST'])
def upload_necklace():
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    raw_bytes = file.read()

    try:
        from rembg import remove
        output_bytes = remove(raw_bytes)
    except Exception as e:
        return jsonify({
            "success": False,
            "error": f"Background removal failed: {e}. "
                     "First use requires internet access to download the rembg model."
        }), 500

    arr = np.frombuffer(output_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None or img.shape[2] != 4:
        return jsonify({"success": False, "error": "Could not process image"}), 400

    img = crop_to_content(img)
    set_uploaded_necklace(img)
    return jsonify({"success": True})


def _process_clothing_upload(file):
    """Shared by /upload_clothing and /upload_clothing_2 — background-removes
    and crops the upload for use as an AI-enhance reference image, and
    separately judges an undertone-compatibility match verdict for it, same
    pattern as lipstick/foundation. The two use the image bytes independently
    (rembg runs twice, once here for a clean AI-reference cutout via
    crop_to_content, once inside extract_product_color for its own
    dominant-color pixel filtering) — a little redundant, but keeps this a
    straightforward reuse of the exact same extract_product_color/
    judge_clothing_match functions the other product routes already use,
    rather than a parallel hand-rolled color-extraction path.

    Returns (rgba_image_or_None, (response_dict, status_code))."""
    raw_bytes = file.read()

    try:
        from rembg import remove
        output_bytes = remove(raw_bytes)
    except Exception as e:
        return None, ({
            "success": False,
            "error": f"Background removal failed: {e}. "
                     "First use requires internet access to download the rembg model."
        }, 500)

    arr = np.frombuffer(output_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_UNCHANGED)

    if img is None or img.shape[2] != 4:
        return None, ({"success": False, "error": "Could not process image"}, 400)

    img = crop_to_content(img)

    match = None
    match_note = None
    try:
        color_result = extract_product_color(raw_bytes)
    except RuntimeError:
        color_result = None

    if color_result is None:
        match_note = "Added for try-on — couldn't isolate a clear garment color to judge the undertone match."
    else:
        _, rgb = color_result
        _, skin_undertone = current_skin_tone()
        if not skin_undertone:
            match_note = "Added for try-on — face isn't detected yet, so we can't judge the undertone match until you're in frame."
        else:
            match = judge_clothing_match(rgb, skin_undertone)

    return img, ({"success": True, "match": match, "match_note": match_note}, 200)


@app.route('/upload_clothing', methods=['POST'])
def upload_clothing():
    """Clothing slot 1 — also the reference image used by the shared
    Jewelry+Clothes 'Capture & Enhance with AI' combined flow (/enhance_photo,
    /enhance_status), unchanged by the Compare Side by Side feature."""
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    img, (payload, status) = _process_clothing_upload(file)
    if img is not None:
        set_uploaded_clothing(img)
    return jsonify(payload), status


@app.route('/upload_clothing_2', methods=['POST'])
def upload_clothing_2():
    """Clothing slot 2 — only used by the Clothes page's Compare Side by
    Side feature (/compare_clothing), independent of slot 1."""
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    img, (payload, status) = _process_clothing_upload(file)
    if img is not None:
        set_uploaded_clothing_2(img)
    return jsonify(payload), status


@app.route('/compare_clothing', methods=['POST'])
def compare_clothing():
    """Runs enhance_captured_photo() once per clothing slot against the same
    base photo, so the user can directly compare both garments on
    themselves. Two separate OpenAI calls (2 credits) — the frontend warns
    about this before the button is clicked. Returns both results as
    base64-encoded JPEGs in one JSON response, since a single Flask response
    can't carry two independent images otherwise."""
    if active_photo["bgr"] is not None:
        photo_bgr = active_photo["bgr"].copy()
    elif latest_raw_frame["bgr"] is not None:
        photo_bgr = latest_raw_frame["bgr"].copy()
    else:
        return jsonify({"success": False, "error": "No live camera frame available yet — make sure the mirror is running."}), 503

    if uploaded_clothing["rgba"] is None or uploaded_clothing_2["rgba"] is None:
        return jsonify({"success": False, "error": "Upload a garment into both slots first."}), 400

    images_b64 = []
    for rgba in (uploaded_clothing["rgba"], uploaded_clothing_2["rgba"]):
        result_bgr, error = enhance_captured_photo(photo_bgr, clothing_rgba=rgba)
        if error:
            return jsonify({"success": False, "error": error}), 400
        ok, buffer = cv2.imencode('.jpg', result_bgr)
        if not ok:
            return jsonify({"success": False, "error": "Failed to encode a result image."}), 500
        images_b64.append(base64.b64encode(buffer.tobytes()).decode('ascii'))

    return jsonify({"success": True, "images": images_b64})


@app.route('/enhance_status')
def enhance_status():
    """Reports which reference images are actually held server-side right
    now, so the frontend can show an accurate 'Will add: ...' summary and
    enable/disable the Enhance with AI button on the Jewelry/Clothes
    pages, rather than tracking upload state independently in JS (which
    could drift out of sync, e.g. after a page reload)."""
    return jsonify({
        "earring": uploaded_earring["rgba"] is not None,
        "necklace": uploaded_necklace["rgba"] is not None,
        "clothing": uploaded_clothing["rgba"] is not None,
    })


@app.route('/enhance_photo', methods=['POST'])
def enhance_photo():
    """Sends a photo plus whichever of earring/necklace/clothing were
    uploaded to OpenAI for a single combined photorealistic AI render —
    the 'Capture & Enhance with AI' button's entire flow in one call.
    Uses active_photo if the user explicitly picked one (Upload Your Photo
    / Use Live Capture), otherwise grabs a fresh live frame automatically
    so the button still works as a single click with no setup. Capture-
    only by design (never runs against the live video stream) — see
    ai_enhance.py's module docstring."""
    if active_photo["bgr"] is not None:
        photo_bgr = active_photo["bgr"].copy()
    elif latest_raw_frame["bgr"] is not None:
        photo_bgr = latest_raw_frame["bgr"].copy()
    else:
        return jsonify({"success": False, "error": "No live camera frame available yet — make sure the mirror is running."}), 503

    result_bgr, error = enhance_captured_photo(
        photo_bgr,
        earring_rgba=uploaded_earring["rgba"],
        necklace_rgba=uploaded_necklace["rgba"],
        clothing_rgba=uploaded_clothing["rgba"],
    )
    if error:
        return jsonify({"success": False, "error": error}), 400

    ok, buffer = cv2.imencode('.jpg', result_bgr)
    if not ok:
        return jsonify({"success": False, "error": "Failed to encode the enhanced image."}), 500

    return Response(buffer.tobytes(), mimetype='image/jpeg')


@app.route('/upload_lipstick_product', methods=['POST'])
def upload_lipstick_product():
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    raw_bytes = file.read()

    try:
        result = extract_product_color(raw_bytes)
    except RuntimeError as e:
        return jsonify({
            "success": False,
            "error": f"{e} First use requires internet access to download the rembg model."
        }), 500

    if result is None:
        return jsonify({"success": False, "error": "Couldn't isolate a clear product color from that photo. Try a closer, more clearly-lit shot."}), 400

    hex_code, rgb = result

    # Apply it live as the current lipstick color
    current_lipstick['hex'] = hex_code

    skin_hex, skin_undertone = current_skin_tone()
    if not skin_undertone:
        match = None
        match_note = "Applied the shade — face isn't detected yet, so we can't judge the match until you're in frame."
    else:
        skin_rgb = None
        if skin_hex:
            skin_rgb = tuple(int(skin_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        match = judge_lipstick_match(rgb, skin_undertone, skin_rgb=skin_rgb)
        match_note = None

    return jsonify({
        "success": True,
        "hex": hex_code,
        "match": match,
        "match_note": match_note
    })


@app.route('/upload_foundation_product', methods=['POST'])
def upload_foundation_product():
    file = request.files.get('image')
    if not file:
        return jsonify({"success": False, "error": "no file provided"}), 400

    raw_bytes = file.read()

    try:
        result = extract_product_color(raw_bytes)
    except RuntimeError as e:
        return jsonify({
            "success": False,
            "error": f"{e} First use requires internet access to download the rembg model."
        }), 500

    if result is None:
        return jsonify({"success": False, "error": "Couldn't isolate a clear product color from that photo. Try a closer, more clearly-lit shot."}), 400

    hex_code, rgb = result

    # Apply it live as the current foundation color
    current_foundation['hex'] = hex_code
    current_foundation['enabled'] = True

    skin_hex, _ = current_skin_tone()
    if not skin_hex:
        match = None
        match_note = "Applied the shade — face isn't detected yet, so we can't judge the match until you're in frame."
    else:
        # convert stored skin hex back to rgb for comparison
        skin_rgb = tuple(int(skin_hex.lstrip('#')[i:i+2], 16) for i in (0, 2, 4))
        match = judge_foundation_match(rgb, skin_rgb)
        match_note = None

    return jsonify({
        "success": True,
        "hex": hex_code,
        "match": match,
        "match_note": match_note
    })


@app.route('/set_foundation', methods=['POST'])
def set_foundation():
    """Mirrors /set_lipstick — lets an already-uploaded foundation slot on
    the Makeup page be re-applied to the live mirror instantly, without
    re-running extraction, when the user taps between slots to compare."""
    data = request.get_json()
    hex_code = data.get('hex')
    if hex_code and len(hex_code) == 7 and hex_code.startswith('#'):
        current_foundation['hex'] = hex_code
        current_foundation['enabled'] = True
        return jsonify({"success": True, "hex": hex_code})
    return jsonify({"success": False, "error": "invalid hex"}), 400


@app.route('/clear_foundation', methods=['POST'])
def clear_foundation():
    current_foundation['enabled'] = False
    current_foundation['hex'] = None
    return jsonify({"success": True})


if __name__ == '__main__':
    print("Starting Glam AI Mirror — open http://127.0.0.1:5000 in your browser")
    # threaded=True is important: /video_feed streams continuously, and
    # without threading it can block other requests (like uploads) from
    # being handled while the video stream is active
    app.run(debug=True, threaded=True)
