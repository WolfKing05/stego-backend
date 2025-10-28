import os
import traceback
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import numpy as np
import cv2
import pywt
import scipy.fftpack as fft
from PIL import Image, ImageOps

app = Flask(__name__)
CORS(app)

ALPHA = 0.02
MAX_DIM = 1024


def apply_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')


def apply_idct(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')


def pil_to_bgr(file):
    img = Image.open(file)
    img = ImageOps.exif_transpose(img).convert("RGB")
    arr = np.array(img)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def bgr_to_png(bgr):
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


# ----------------- ENCODE -----------------
def encode_dct_dwt_color(cover, secret):
    h, w = cover.shape[:2]
    secret = cv2.resize(secret, (w, h))
    stego_channels = []

    for ch in range(3):
        c = cover[:, :, ch].astype(np.float32)
        s = secret[:, :, ch].astype(np.float32)

        # 1-level Haar DWT
        cA, (cH, cV, cD) = pywt.dwt2(c, 'haar')
        sA, _ = pywt.dwt2(s, 'haar')

        # DCT on approximation only
        cA_dct = apply_dct(cA)
        sA_dct = apply_dct(sA)

        # Embed secret energy into cover approximation
        embedded = cA_dct + ALPHA * sA_dct

        # IDCT + inverse DWT
        cA_embedded = apply_idct(embedded)
        stego = pywt.idwt2((cA_embedded, (cH, cV, cD)), 'haar')
        stego = np.clip(stego, 0, 255).astype(np.uint8)
        stego_channels.append(stego)

    stego_img = cv2.merge(stego_channels)
    return stego_img


# ----------------- DECODE -----------------
def decode_dct_dwt_color(stego, cover):
    h, w = stego.shape[:2]
    cover = cv2.resize(cover, (w, h))
    extracted_channels = []

    for ch in range(3):
        s = stego[:, :, ch].astype(np.float32)
        c = cover[:, :, ch].astype(np.float32)

        sA, _ = pywt.dwt2(s, 'haar')
        cA, _ = pywt.dwt2(c, 'haar')

        sA_dct = apply_dct(sA)
        cA_dct = apply_dct(cA)

        # Extract secret coefficients ONLY from approximation energy
        extracted_dct = (sA_dct - cA_dct) / ALPHA
        extracted = apply_idct(extracted_dct)

        # Normalize to full intensity range
        extracted = cv2.normalize(extracted, None, 0, 255, cv2.NORM_MINMAX)
        extracted = np.clip(extracted, 0, 255).astype(np.uint8)

        # Median filter to suppress cover edges
        extracted = cv2.medianBlur(extracted, 3)

        extracted_channels.append(extracted)

    secret = cv2.merge(extracted_channels)
    secret = cv2.normalize(secret, None, 0, 255, cv2.NORM_MINMAX)
    return secret


@app.route("/")
def home():
    return jsonify({"status": "ok"}), 200


@app.route("/encode", methods=["POST"])
def encode_route():
    try:
        if "cover" not in request.files or "secret" not in request.files:
            return jsonify({"error": "Missing images"}), 400

        cover = pil_to_bgr(request.files["cover"])
        secret = pil_to_bgr(request.files["secret"])

        if max(cover.shape[:2]) > MAX_DIM:
            scale = MAX_DIM / max(cover.shape[:2])
            new_w = int(cover.shape[1] * scale)
            new_h = int(cover.shape[0] * scale)
            cover = cv2.resize(cover, (new_w, new_h))
            secret = cv2.resize(secret, (new_w, new_h))

        stego = encode_dct_dwt_color(cover, secret)
        buf = bgr_to_png(stego)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="stego.png")

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return tb, 500


@app.route("/decode", methods=["POST"])
def decode_route():
    try:
        if "stego" not in request.files or "cover" not in request.files:
            return jsonify({"error": "Missing images"}), 400

        stego = pil_to_bgr(request.files["stego"])
        cover = pil_to_bgr(request.files["cover"])

        secret = decode_dct_dwt_color(stego, cover)
        buf = bgr_to_png(secret)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="extracted.png")

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return tb, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
