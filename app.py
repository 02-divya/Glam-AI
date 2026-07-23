from flask import Flask, render_template, Response, jsonify, request
import numpy as np
import cv2
from camera import (generate_frames, latest_status, latest_skin_tone,
                     current_lipstick, current_jewelry, set_uploaded_earring,
                     set_uploaded_necklace, current_necklace)

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


if __name__ == '__main__':
    print("Starting Glam AI Mirror — open http://127.0.0.1:5000 in your browser")
    # threaded=True is important: /video_feed streams continuously, and
    # without threading it can block other requests (like uploads) from
    # being handled while the video stream is active
    app.run(debug=True, threaded=True)