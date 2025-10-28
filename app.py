# app.py - Robust LSB steganography backend with auto-detect header
import os
import traceback
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)
CORS(app)

# Config
BITS_PER_CHANNEL = 1     # LSBs per color channel used (1 is safest)
HEADER_BITS = 32         # 16 bits width + 16 bits height
MAX_DIM = 2048           # max cover dimension allowed for processing


# Helpers
def pil_open_fix(fileobj):
    img = Image.open(fileobj)
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def pil_to_rgb_np(img):
    # Return HxWx3 uint8 RGB (composite alpha over white if present)
    if img.mode == "RGBA":
        bg = Image.new("RGBA", img.size, (255,255,255,255))
        img = Image.alpha_composite(bg, img).convert("RGB")
    else:
        img = img.convert("RGB")
    arr = np.array(img, dtype=np.uint8)
    return arr

def np_to_png_bytes(arr):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def int_to_bits(value, bitcount):
    return [(value >> (bitcount-1-i)) & 1 for i in range(bitcount)]

def bits_to_int(bits):
    v = 0
    for b in bits:
        v = (v << 1) | int(b)
    return v

# Capacity calculation
def capacity_bits_for_cover(cover_arr):
    H, W = cover_arr.shape[:2]
    return H * W * 3 * BITS_PER_CHANNEL

# Embed
def lsb_embed_with_header(cover_arr, secret_arr):
    Hc, Wc = cover_arr.shape[:2]
    Hs, Ws = secret_arr.shape[:2]
    secret_bytes = secret_arr.reshape(-1)  # flattened bytes (R,G,B)...
    secret_bits = np.unpackbits(secret_bytes)
    header_bits = int_to_bits(Ws, 16) + int_to_bits(Hs, 16)  # width(16) then height(16)
    full_bits = np.array(header_bits + secret_bits.tolist(), dtype=np.uint8)

    cap = capacity_bits_for_cover(cover_arr)
    if full_bits.size > cap:
        raise ValueError("Secret too large for cover using current settings")

    out = cover_arr.copy().reshape(-1)  # flatten bytes
    bit_idx = 0
    for i in range(out.size):
        if bit_idx >= full_bits.size:
            break
        # clear LSB(s)
        mask = ~((1 << BITS_PER_CHANNEL) - 1) & 0xFF
        out[i] = out[i] & mask
        # pack next BITS_PER_CHANNEL bits into val
        val = 0
        for b in range(BITS_PER_CHANNEL):
            if bit_idx < full_bits.size:
                val = (val << 1) | int(full_bits[bit_idx])
                bit_idx += 1
            else:
                val = (val << 1)
        out[i] = out[i] | val

    out_arr = out.reshape(cover_arr.shape)
    return out_arr

# Extract
def lsb_extract_autodetect(stego_arr):
    flat = stego_arr.reshape(-1)
    bits = []
    # Read enough bits to get header first
    needed_for_header = HEADER_BITS
    i = 0
    while len(bits) < needed_for_header and i < flat.size:
        byte = int(flat[i])
        for b in reversed(range(BITS_PER_CHANNEL)):
            bits.append((byte >> b) & 1)
            if len(bits) >= needed_for_header:
                break
        i += 1
    if len(bits) < needed_for_header:
        raise ValueError("Not enough data to read header")

    header_width = bits_to_int(bits[:16])
    header_height = bits_to_int(bits[16:32])
    if header_width == 0 or header_height == 0:
        raise ValueError("Invalid header dimensions detected")

    secret_pixel_count = header_width * header_height * 3
    secret_bits_needed = secret_pixel_count * 8
    total_bits_needed = HEADER_BITS + secret_bits_needed

    # continue reading remaining bits
    while len(bits) < total_bits_needed and i < flat.size:
        byte = int(flat[i])
        for b in reversed(range(BITS_PER_CHANNEL)):
            bits.append((byte >> b) & 1)
            if len(bits) >= total_bits_needed:
                break
        i += 1

    if len(bits) < total_bits_needed:
        raise ValueError("Not enough embedded bits for full secret image")

    secret_bits = np.array(bits[HEADER_BITS:HEADER_BITS+secret_bits_needed], dtype=np.uint8)
    secret_bytes = np.packbits(secret_bits)
    secret_arr = secret_bytes.reshape((header_height, header_width, 3)).astype(np.uint8)
    return secret_arr

# Auto-resize secret to fit capacity if needed (while preserving aspect)
def ensure_secret_fits(cover_arr, secret_img_arr):
    cap_bits = capacity_bits_for_cover(cover_arr)
    max_secret_bytes = (cap_bits - HEADER_BITS) // 8
    if max_secret_bytes <= 0:
        raise ValueError("Cover too small to embed any secret")
    cur_bytes = secret_img_arr.size
    if cur_bytes <= max_secret_bytes:
        return secret_img_arr
    # compute scaling factor
    Hs, Ws = secret_img_arr.shape[:2]
    cur_pixels = Hs * Ws
    max_pixels = max_secret_bytes // 3
    scale = (max_pixels / cur_pixels) ** 0.5
    new_w = max(1, int(Ws * scale))
    new_h = max(1, int(Hs * scale))
    from PIL import Image
    pil = Image.fromarray(secret_img_arr)
    pil2 = pil.resize((new_w, new_h), Image.LANCZOS)
    return np.array(pil2, dtype=np.uint8)

# Routes
@app.route("/")
def index():
    return jsonify({"service":"LSB Stego API","status":"ok"}), 200

@app.route("/encode", methods=["POST"])
def encode_route():
    try:
        if "cover" not in request.files or "secret" not in request.files:
            return jsonify({"error":"upload 'cover' and 'secret' files"}), 400

        cover_img = pil_open_fix(request.files["cover"])
        secret_img = pil_open_fix(request.files["secret"])

        # limit cover size for safety
        if max(cover_img.size) > MAX_DIM:
            scale = MAX_DIM / max(cover_img.size)
            cover_img = cover_img.resize((int(cover_img.width*scale), int(cover_img.height*scale)), Image.LANCZOS)

        cover_arr = pil_to_rgb_np(cover_img)
        secret_arr = pil_to_rgb_np(secret_img)

        # auto-resize secret if needed
        secret_arr = ensure_secret_fits(cover_arr, secret_arr)

        stego_arr = lsb_embed_with_header(cover_arr, secret_arr)
        buf = np_to_png_bytes(stego_arr)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="stego.png")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "trace": tb}), 500

@app.route("/decode", methods=["POST"])
def decode_route():
    try:
        if "stego" not in request.files:
            return jsonify({"error":"upload 'stego' file"}), 400

        stego_img = pil_open_fix(request.files["stego"])
        stego_arr = pil_to_rgb_np(stego_img)

        secret_arr = lsb_extract_autodetect(stego_arr)
        buf = np_to_png_bytes(secret_arr)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="extracted.png")
    except Exception as e:
        tb = traceback.format_exc()
        print(tb)
        return jsonify({"error": str(e), "trace": tb}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
