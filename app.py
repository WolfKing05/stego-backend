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

# ======================== CONFIG ========================
MAX_DIM = 1024
ALPHA = 0.02  # smaller = less distortion, better separation
# ========================================================


def apply_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')


def apply_idct(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')


def pil_to_bgr(pil_img):
    pil_img = ImageOps.exif_transpose(pil_img).convert("RGB")
    arr = np.array(pil_img)
    arr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return arr


def bgr_to_png_bytes(bgr_arr):
    rgb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    buf = BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


# =================== CORE ALGORITHMS ===================

def encode_dct_dwt_color(cover, secret):
    h, w = cover.shape[:2]
    secret = cv2.resize(secret, (w, h))
    stego_channels = []

    for ch in range(3):
        c_ch = cover[:, :, ch].astype(np.float32)
        s_ch = secret[:, :, ch].astype(np.float32)

        cA, (cH, cV, cD) = pywt.dwt2(c_ch, 'haar')
        sA, _ = pywt.dwt2(s_ch, 'haar')

        cA_dct = apply_dct(cA)
        sA_dct = apply_dct(sA)

        embedded_dct = cA_dct + ALPHA * sA_dct
        cA_embedded = apply_idct(embedded_dct)
        stego_ch = pywt.idwt2((cA_embedded, (cH, cV, cD)), 'haar')
        stego_ch = np.clip(stego_ch, 0, 255).astype(np.uint8)

        stego_channels.append(stego_ch)

    stego = cv2.merge(stego_channels)
    return stego


def decode_dct_dwt_color(stego, cover):
    h, w = stego.shape[:2]
    cover = cv2.resize(cover, (w, h))
    extracted_channels = []

    for ch in range(3):
        s_ch = stego[:, :, ch].astype(np.float32)
        c_ch = cover[:, :, ch].astype(np.float32)

        sA, _ = pywt.dwt2(s_ch, 'haar')
        cA, _ = pywt.dwt2(c_ch, 'haar')

        sA_dct = apply_dct(sA)
        cA_dct = apply_dct(cA)

        diff_dct = (sA_dct - cA_dct) / ALPHA
        recovered = apply_idct(diff_dct)
        recovered = np.clip(recovered, 0, 255).astype(np.uint8)

        # Contrast normalize per channel to remove cover bleed
        recovered = cv2.normalize(recovered, None, 0, 255, cv2.NORM_MINMAX)
        extracted_channels.append(recovered)

    extracted = cv2.merge(extracted_channels)
    extracted = cv2.medianBlur(extracted, 3)
    extracted = cv2.normalize(extracted, None, 0, 255, cv2.NORM_MINMAX)
    return extracted


# =================== ROUTES ===================

@app.route("/")
def home():
    return jsonify({"status": "ok", "info": "Image Steganography API"}), 200


@app.route("/encode", methods=["POST"])
def encode_route():
    try:
        if "cover" not in request.files or "secret" not in request.files:
            return jsonify({"error": "Missing cover or secret image"}), 400

        cover = pil_to_bgr(Image.open(request.files["cover"]))
        secret = pil_to_bgr(Image.open(request.files["secret"]))

        if max(cover.shape[:2]) > MAX_DIM:
            scale = MAX_DIM / max(cover.shape[:2])
            cover = cv2.resize(cover, (int(cover.shape[1] * scale), int(cover.shape[0] * scale)))
            secret = cv2.resize(secret, (cover.shape[1], cover.shape[0]))

        stego = encode_dct_dwt_color(cover, secret)
        buf = bgr_to_png_bytes(stego)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="stego.png")

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return tb, 500


@app.route("/decode", methods=["POST"])
def decode_route():
    try:
        if "stego" not in request.files or "cover" not in request.files:
            return jsonify({"error": "Missing stego or cover image"}), 400

        stego = pil_to_bgr(Image.open(request.files["stego"]))
        cover = pil_to_bgr(Image.open(request.files["cover"]))

        extracted = decode_dct_dwt_color(stego, cover)
        buf = bgr_to_png_bytes(extracted)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="extracted.png")

    except Exception:
        tb = traceback.format_exc()
        print(tb)
        return tb, 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 8080)))
