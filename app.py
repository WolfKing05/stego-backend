# app.py
from flask import Flask, request, send_file, jsonify
import numpy as np
import cv2
import pywt
import scipy.fftpack as fft
from PIL import Image, ImageOps
from io import BytesIO
import os

from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# ---------------- Utility functions (DCT/DWT) ----------------

def apply_dct(block):
    return fft.dct(fft.dct(block.T, norm='ortho').T, norm='ortho')

def apply_idct(block):
    return fft.idct(fft.idct(block.T, norm='ortho').T, norm='ortho')

# ---------------- Adaptive encode/decode (array-based) ----------------

def encode_dct_dwt_color_array(cover_arr, secret_arr, base_alpha=0.05, max_dim=1024):
    """
    Adaptive embedding: returns uint8 BGR stego image.
    base_alpha: nominal factor (0.03-0.08 typical). Function adapts per-channel.
    """
    h, w = cover_arr.shape[:2]
    if max(h, w) > max_dim:
        scale = max_dim / max(h, w)
        cover_arr = cv2.resize(cover_arr, (int(w*scale), int(h*scale)))
        secret_arr = cv2.resize(secret_arr, (cover_arr.shape[1], cover_arr.shape[0]))

    stego_channels = []
    for ch in range(3):
        cover_ch = cover_arr[:, :, ch].astype(np.float32)
        secret_ch = secret_arr[:, :, ch].astype(np.float32)

        # DWT
        cA, (cH, cV, cD) = pywt.dwt2(cover_ch, 'haar')
        sA, (_, _, _) = pywt.dwt2(secret_ch, 'haar')

        # DCT on approximation coefficients
        cA_dct = apply_dct(cA)
        sA_dct = apply_dct(sA)

        # adaptive alpha: scale secret relative to cover energy
        cover_energy = (np.std(cA_dct) + 1e-8)
        secret_energy = (np.std(sA_dct) + 1e-8)
        adapt_factor = (cover_energy / secret_energy)
        alpha = base_alpha * adapt_factor
        alpha = float(np.clip(alpha, 0.005, 0.12))

        embedded_dct = cA_dct + alpha * sA_dct
        cA_embedded = apply_idct(embedded_dct)

        stego_ch = pywt.idwt2((cA_embedded, (cH, cV, cD)), 'haar')
        stego_ch = np.clip(stego_ch, 0, 255).astype(np.uint8)

        stego_channels.append(stego_ch)

    stego_image = cv2.merge(stego_channels)
    return stego_image

def decode_dct_dwt_color_array(stego_arr, cover_arr, base_alpha=0.05):
    """
    Adaptive decoding: returns uint8 BGR extracted image (normalized + mild denoise).
    """
    extracted_channels = []
    for ch in range(3):
        stego_ch = stego_arr[:, :, ch].astype(np.float32)
        cover_ch = cover_arr[:, :, ch].astype(np.float32)

        sA, _ = pywt.dwt2(stego_ch, 'haar')
        cA, _ = pywt.dwt2(cover_ch, 'haar')

        sA_dct = apply_dct(sA)
        cA_dct = apply_dct(cA)

        cover_energy = (np.std(cA_dct) + 1e-8)
        # for secret_energy in decode, estimate using sA_dct magnitude
        secret_energy = (np.std(sA_dct) + 1e-8)
        adapt_factor = (cover_energy / secret_energy)
        alpha = base_alpha * adapt_factor
        alpha = float(np.clip(alpha, 0.005, 0.12))

        extracted_dct = (sA_dct - cA_dct) / (alpha + 1e-12)
        extracted = apply_idct(extracted_dct)

        extracted = np.clip(extracted, 0, 255).astype(np.uint8)

        # small median blur to reduce speckle
        try:
            extracted = cv2.medianBlur(extracted, 3)
        except Exception:
            pass

        extracted_channels.append(extracted)

    extracted_img = cv2.merge(extracted_channels)

    # Contrast stretch to full range for visibility
    extracted_img = cv2.normalize(extracted_img, None, 0, 255, cv2.NORM_MINMAX)

    # optional gamma (disabled by default)
    gamma = 1.0
    if abs(gamma - 1.0) > 1e-6:
        invGamma = 1.0 / gamma
        table = (np.array([((i / 255.0) ** invGamma) * 255 for i in range(256)])).astype("uint8")
        extracted_img = cv2.LUT(extracted_img.astype(np.uint8), table)

    return extracted_img.astype(np.uint8)

# ---------------- Helpers for file handling ----------------

def pil_image_from_upload(file_storage):
    """
    Read a Flask uploaded file and return a PIL Image with EXIF orientation applied.
    """
    img = Image.open(file_storage.stream)
    img = ImageOps.exif_transpose(img)
    return img.convert("RGB")

def pil_to_bgr_np(pil_img):
    """
    Convert PIL RGB image to OpenCV BGR numpy array (uint8).
    """
    rgb = np.array(pil_img)
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

# ---------------- Flask routes ----------------

@app.route("/")
def index():
    return "Image Steganography API (DWT+DCT). POST /encode or /decode"

@app.route("/encode", methods=["POST"])
def encode_route():
    if "cover" not in request.files or "secret" not in request.files:
        return jsonify({"error": "Please upload 'cover' and 'secret' files."}), 400

    try:
        cover_pil = pil_image_from_upload(request.files["cover"])
        secret_pil = pil_image_from_upload(request.files["secret"])

        cover_arr = pil_to_bgr_np(cover_pil)
        secret_arr = pil_to_bgr_np(secret_pil)

        # ensure reasonable size for speed
        max_dim = 1024
        h, w = cover_arr.shape[:2]
        if max(h, w) > max_dim:
            scale = max_dim / max(h, w)
            new_w = int(w * scale)
            new_h = int(h * scale)
            cover_arr = cv2.resize(cover_arr, (new_w, new_h))
            secret_arr = cv2.resize(secret_arr, (new_w, new_h))

        # embed with adaptive base alpha
        base_alpha = 0.05
        stego_arr = encode_dct_dwt_color_array(cover_arr, secret_arr, base_alpha=base_alpha)

        mem = bgr_np_to_png_bytes(stego_arr)
        return send_file(mem, mimetype="image/png", as_attachment=True, download_name="stego.png")
    except Exception as e:
        # for debugging return the error message (Render logs will contain trace)
        return str(e), 500

@app.route("/decode", methods=["POST"])
def decode_route():
    if "stego" not in request.files or "cover" not in request.files:
        return jsonify({"error": "Please upload 'stego' and 'cover' files."}), 400

    try:
        stego_pil = pil_image_from_upload(request.files["stego"])
        cover_pil = pil_image_from_upload(request.files["cover"])

        stego_arr = pil_to_bgr_np(stego_pil)
        cover_arr = pil_to_bgr_np(cover_pil)

        # make sure sizes match (resize cover to stego if needed)
        if stego_arr.shape[:2] != cover_arr.shape[:2]:
            cover_arr = cv2.resize(cover_arr, (stego_arr.shape[1], stego_arr.shape[0]))

        base_alpha = 0.05
        extracted_arr = decode_dct_dwt_color_array(stego_arr, cover_arr, base_alpha=base_alpha)

        mem = bgr_np_to_png_bytes(extracted_arr)
        return send_file(mem, mimetype="image/png", as_attachment=True, download_name="extracted.png")
    except Exception as e:
        return str(e), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)
