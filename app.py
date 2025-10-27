from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import pywt
import scipy.fftpack as fft
from PIL import Image
from io import BytesIO
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def apply_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')

def encode_dct_dwt_color_array(cover_arr, secret_arr, alpha=0.01, max_dim=512):
    h, w = cover_arr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        cover_arr = cv2.resize(cover_arr, (new_w, new_h))
        secret_arr = cv2.resize(secret_arr, (new_w, new_h))
    secret_img = cv2.resize(secret_arr, (cover_arr.shape[1], cover_arr.shape[0]))
    stego_channels = []
    for ch in range(3):
        cover_ch = cover_arr[:, :, ch].astype(np.float32)
        secret_ch = secret_img[:, :, ch].astype(np.float32)
        cA, (cH, cV, cD) = pywt.dwt2(cover_ch, 'haar')
        sA, (_, _, _) = pywt.dwt2(secret_ch, 'haar')
        cA_dct = apply_dct(cA)
        sA_dct = apply_dct(sA)
        embedded_dct = cA_dct + alpha * sA_dct
        cA_embedded = apply_idct(embedded_dct)
        stego_ch = pywt.idwt2((cA_embedded, (cH, cV, cD)), 'haar')
        stego_ch = np.clip(stego_ch, 0, 255).astype(np.uint8)
        stego_channels.append(stego_ch)
    stego_image = cv2.merge(stego_channels)
    return stego_image

def decode_dct_dwt_color_array(stego_arr, cover_arr, alpha=0.01):
    extracted_channels = []
    for ch in range(3):
        stego_ch = stego_arr[:, :, ch].astype(np.float32)
        cover_ch = cover_arr[:, :, ch].astype(np.float32)
        sA, _ = pywt.dwt2(stego_ch, 'haar')
        cA, _ = pywt.dwt2(cover_ch, 'haar')
        sA_dct = apply_dct(sA)
        cA_dct = apply_dct(cA)
        extracted_dct = (sA_dct - cA_dct) / alpha
        extracted = apply_idct(extracted_dct)
        extracted = np.clip(extracted, 0, 255).astype(np.uint8)
        extracted_channels.append(extracted)
    extracted_img = cv2.merge(extracted_channels)
    return extracted_img

@app.route("/")
def index():
    return "Image Steganography API (DWT+DCT). POST /encode or /decode"

@app.route("/encode", methods=["POST"])
def encode_route():
    if "cover" not in request.files or "secret" not in request.files:
        return jsonify({"error":"Please upload 'cover' and 'secret' files."}), 400
    cover = Image.open(request.files["cover"]).convert("RGB")
    secret = Image.open(request.files["secret"]).convert("RGB")
    cover_arr = cv2.cvtColor(np.array(cover), cv2.COLOR_RGB2BGR)
    secret_arr = cv2.cvtColor(np.array(secret), cv2.COLOR_RGB2BGR)
    try:
        stego = encode_dct_dwt_color_array(cover_arr, secret_arr)
    except Exception as e:
        return str(e), 500
    stego_rgb = cv2.cvtColor(stego, cv2.COLOR_BGR2RGB)
    mem = BytesIO()
    Image.fromarray(stego_rgb).save(mem, format='PNG')
    mem.seek(0)
    return send_file(mem, mimetype='image/png', as_attachment=True, download_name='stego.png')

@app.route("/decode", methods=["POST"])
def decode_route():
    if "stego" not in request.files or "cover" not in request.files:
        return jsonify({"error":"Please upload 'stego' and 'cover' files."}), 400
    stego = Image.open(request.files["stego"]).convert("RGB")
    cover = Image.open(request.files["cover"]).convert("RGB")
    stego_arr = cv2.cvtColor(np.array(stego), cv2.COLOR_RGB2BGR)
    cover_arr = cv2.cvtColor(np.array(cover), cv2.COLOR_RGB2BGR)
    try:
        extracted = decode_dct_dwt_color_array(stego_arr, cover_arr)
    except Exception as e:
        return str(e), 500
    extracted_rgb = cv2.cvtColor(extracted, cv2.COLOR_BGR2RGB)
    mem = BytesIO()
    Image.fromarray(extracted_rgb).save(mem, format='PNG')
    mem.seek(0)
    return send_file(mem, mimetype='image/png', as_attachment=True, download_name='extracted.png')

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
