# app.py — robust final backend for Image Steganography (DWT + DCT)
# Paste/replace this file in your stego-backend repo and deploy.

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
CORS(app, resources={r"/*": {"origins": "*"}})

# ------- Configurable parameters (tweak only these if needed) -------
MAX_DIM = int(os.environ.get("MAX_DIM", 1024))      # max image dimension server-side
BASE_ALPHA = float(os.environ.get("BASE_ALPHA", 0.03))  # base_alpha for adaptive embedding
MIN_ALPHA = 0.005
MAX_ALPHA = 0.14

# ------- Utility: DCT / IDCT wrappers (same as desktop) -------
def apply_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')

# ------- Adaptive encode / decode using arrays -------
def encode_dct_dwt_color_array(cover_arr, secret_arr, base_alpha=BASE_ALPHA, max_dim=MAX_DIM):
    """
    cover_arr, secret_arr: BGR uint8 numpy arrays
    Returns: stego_arr (uint8 BGR)
    """
    # ensure reasonable size for performance/stability
    h, w = cover_arr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        new_w, new_h = int(w * scale), int(h * scale)
        cover_arr = cv2.resize(cover_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
        secret_arr = cv2.resize(secret_arr, (new_w, new_h), interpolation=cv2.INTER_AREA)
    else:
        # ensure secret matches cover size
        secret_arr = cv2.resize(secret_arr, (w, h), interpolation=cv2.INTER_AREA)

    stego_channels = []
    for ch in range(3):
        cover_ch = cover_arr[:, :, ch].astype(np.float32)
        secret_ch = secret_arr[:, :, ch].astype(np.float32)

        # DWT (approximation + details)
        cA, (cH, cV, cD) = pywt.dwt2(cover_ch, 'haar')
        sA, (_, _, _) = pywt.dwt2(secret_ch, 'haar')

        # DCT on approximation
        cA_dct = apply_dct(cA)
        sA_dct = apply_dct(sA)

        # adaptive alpha based on energy (std)
        cover_energy = (np.std(cA_dct) + 1e-8)
        secret_energy = (np.std(sA_dct) + 1e-8)
        adapt_factor = (cover_energy / secret_energy)
        alpha = base_alpha * adapt_factor
        alpha = float(np.clip(alpha, MIN_ALPHA, MAX_ALPHA))

        # embed
        embedded_dct = cA_dct + alpha * sA_dct
        cA_embedded = apply_idct(embedded_dct)

        # inverse DWT
        stego_ch = pywt.idwt2((cA_embedded, (cH, cV, cD)), 'haar')
        stego_ch = np.clip(stego_ch, 0, 255).astype(np.uint8)
        stego_channels.append(stego_ch)

    stego_image = cv2.merge(stego_channels)
    return stego_image

def decode_dct_dwt_color_array(stego_arr, cover_arr, base_alpha=BASE_ALPHA):
    """
    stego_arr, cover_arr: BGR arrays (uint8)
    Returns extracted_arr (uint8 BGR), contrast stretched and lightly denoised.
    """
    # if shapes differ, resize cover to stego
    if stego_arr.shape[:2] != cover_arr.shape[:2]:
        cover_arr = cv2.resize(cover_arr, (stego_arr.shape[1], stego_arr.shape[0]), interpolation=cv2.INTER_AREA)

    extracted_channels = []
    for ch in range(3):
        stego_ch = stego_arr[:, :, ch].astype(np.float32)
        cover_ch = cover_arr[:, :, ch].astype(np.float32)

        sA, _ = pywt.dwt2(stego_ch, 'haar')
        cA, _ = pywt.dwt2(cover_ch, 'haar')

        sA_dct = apply_dct(sA)
        cA_dct = apply_dct(cA)

        # recompute adaptive alpha same as encode
        cover_energy = (np.std(cA_dct) + 1e-8)
        secret_energy = (np.std(sA_dct) + 1e-8)
        adapt_factor = (cover_energy / secret_energy)
        alpha = base_alpha * adapt_factor
        alpha = float(np.clip(alpha, MIN_ALPHA, MAX_ALPHA))

        # extract
        extracted_dct = (sA_dct - cA_dct) / (alpha + 1e-12)
        extracted = apply_idct(extracted_dct)
        extracted = np.clip(extracted, 0, 255).astype(np.uint8)

        # mild median to remove speckle
        try:
            extracted = cv2.medianBlur(extracted, 3)
        except Exception:
            pass

        extracted_channels.append(extracted)

    extracted_img = cv2.merge(extracted_channels)

    # contrast stretch to full dynamic range
    extracted_img = cv2.normalize(extracted_img, None, 0, 255, cv2.NORM_MINMAX)

    # optional gamma adjustment (disabled by default)
    # gamma = 1.0
    # if abs(gamma - 1.0) > 1e-6:
    #     invGamma = 1.0 / gamma
    #     table = (np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)])).astype("uint8")
    #     extracted_img = cv2.LUT(extracted_img.astype(np.uint8), table)

    return extracted_img.astype(np.uint8)

# ------- Helpers for reading uploads and converting -------
def pil_image_from_upload(file_storage):
    """
    Read a Flask uploaded file and return PIL RGB image with EXIF orientation fixed.
    """
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def pil_to_bgr_np(pil_img):
    """
    Convert PIL RGB image to OpenCV BGR uint8 numpy array.
    """
    rgb = np.array(pil_img)
    # handle single-channel or unexpected shapes
    if rgb.ndim == 2:
        rgb = np.stack([rgb, rgb, rgb], axis=-1)
    if rgb.shape[2] == 4:
        rgb = rgb[:, :, :3]
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    return bgr

def bgr_np_to_png_bytes(bgr_arr):
    """
    Convert OpenCV BGR array to PNG bytes (RGB saved as PNG).
    """
    rgb = cv2.cvtColor(bgr_arr, cv2.COLOR_BGR2RGB)
    pil = Image.fromarray(rgb)
    mem = BytesIO()
    pil.save(mem, format='PNG')
    mem.seek(0)
    return mem

# ------- Routes -------
@app.route("/")
def index():
    return jsonify({"service": "Image Steganography API (DWT+DCT)", "status": "ok"}), 200

@app.route("/encode", methods=["POST"])
def encode_route():
    try:
        if "cover" not in request.files or "secret" not in request.files:
            return jsonify({"error": "Please upload both 'cover' and 'secret' files."}), 400

        cover_pil = pil_image_from_upload(request.files["cover"])
        secret_pil = pil_image_from_upload(request.files["secret"])

        cover_arr = pil_to_bgr_np(cover_pil)
        secret_arr = pil_to_bgr_np(secret_pil)

        # safety: if either dimension is tiny, reject
        if cover_arr.shape[0] < 16 or cover_arr.shape[1] < 16:
            return jsonify({"error": "Cover image too small."}), 400

        # perform encoding
        stego_arr = encode_dct_dwt_color_array(cover_arr, secret_arr, base_alpha=BASE_ALPHA, max_dim=MAX_DIM)

        mem = bgr_np_to_png_bytes(stego_arr)
        return send_file(mem, mimetype="image/png", as_attachment=True, download_name="stego.png")
    except Exception as e:
        # print full traceback to logs and return it in response to aid debugging in short deadline
        tb = traceback.format_exc()
        print(tb)
        return ("ERROR during /encode:\n" + tb), 500

@app.route("/decode", methods=["POST"])
def decode_route():
    try:
        if "stego" not in request.files or "cover" not in request.files:
            return jsonify({"error": "Please upload both 'stego' and 'cover' files."}), 400

        stego_pil = pil_image_from_upload(request.files["stego"])
        cover_pil = pil_image_from_upload(request.files["cover"])

        stego_arr = pil_to_bgr_np(stego_pil)
        cover_arr = pil_to_bgr_np(cover_pil)

        extracted_arr = decode_dct_dwt_color_array(stego_arr, cover_arr, base_alpha=BASE_ALPHA)

        mem = bgr_np_to_png_bytes(extracted_arr)
        return send_file(mem, mimetype="image/png", as_attachment=True, download_name="extracted.png")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return ("ERROR during /decode:\n" + tb), 500

# ------- Run -------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    # In debug mode you can set FLASK_DEBUG=1 but Render uses gunicorn in production
    app.run(host="0.0.0.0", port=port)
