from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import cv2
from camera import (generate_frames, latest_status, latest_skin_tone,
                     current_lipstick, current_jewelry, set_uploaded_earring,
                     set_uploaded_necklace, current_necklace, current_foundation)
from commerce.deal_finder import find_best_deal
from analysis.product_match import extract_product_color, judge_foundation_match, judge_lipstick_match

app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                     mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/status')
def status():
    return jsonify(latest_status)


@app.route('/skin_tone')
def skin_tone():
    return jsonify(latest_skin_tone)


@app.route('/best_deal')
def best_deal():
    undertone = latest_skin_tone.get("undertone")
    product_type = request.args.get("product_type", "lipstick")

    if not undertone:
        return jsonify({"success": False, "error": "No skin tone detected yet — make sure your face is in frame."}), 400

    deal = find_best_deal(undertone, product_type)
    if not deal:
        return jsonify({"success": False, "error": f"No {product_type} matches found for undertone '{undertone}'."}), 404

    return jsonify({"success": True, "undertone": undertone, "deal": deal})


@app.route('/set_lipstick', methods=['POST'])
def set_lipstick():
    data = request.get_json()
    hex_code = data.get('hex')
    if hex_code and len(hex_code) == 7 and hex_code.startswith('#'):
        current_lipstick['hex'] = hex_code
        return jsonify({"success": True, "hex": hex_code})
    return jsonify({"success": False, "error": "invalid hex"}), 400


@app.route('/set_jewelry_style', methods=['POST'])
def set_jewelry_style():
    data = request.get_json()
    style = data.get('style')
    if style in ('stud', 'hoop', 'uploaded', 'off'):
        if style == 'off':
            current_jewelry['enabled'] = False
        else:
            current_jewelry['enabled'] = True
            current_jewelry['style'] = style
        return jsonify({"success": True})
    return jsonify({"success": False, "error": "invalid style"}), 400


@app.route('/adjust_earring', methods=['POST'])
def adjust_earring():
    data = request.get_json()
    if 'scale_adjust' in data:
        current_jewelry['scale_adjust'] = float(data['scale_adjust'])
    if 'horizontal_adjust' in data:
        current_jewelry['horizontal_adjust'] = float(data['horizontal_adjust'])
    if 'vertical_adjust' in data:
        current_jewelry['vertical_adjust'] = float(data['vertical_adjust'])
    return jsonify({"success": True, **current_jewelry})


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

    set_uploaded_necklace(img)
    return jsonify({"success": True})


@app.route('/adjust_necklace', methods=['POST'])
def adjust_necklace():
    data = request.get_json()
    if 'offset_adjust' in data:
        current_necklace['offset_adjust'] = float(data['offset_adjust'])
    if 'scale_adjust' in data:
        current_necklace['scale_adjust'] = float(data['scale_adjust'])
    return jsonify({"success": True, **current_necklace})


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

    skin_undertone = latest_skin_tone.get("undertone")
    if not skin_undertone:
        match = None
        match_note = "Applied the shade — face isn't detected yet, so we can't judge the match until you're in frame."
    else:
        match = judge_lipstick_match(rgb, skin_undertone)
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

    skin_hex = latest_skin_tone.get("hex")
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