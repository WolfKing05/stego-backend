# app.py - Simple robust LSB steganography backend (drop-in replacement)
import os
from io import BytesIO
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
from PIL import Image, ImageOps
import numpy as np

app = Flask(__name__)
CORS(app)

# Limits
MAX_DIM = 2048            # max cover dimension (to avoid huge uploads)
BITS_PER_CHANNEL = 1      # how many LSBs per color channel to use (1 is safest)
CHANNELS = 3              # use RGB channels

def pil_open_fix(fileobj):
    img = Image.open(fileobj)
    img = ImageOps.exif_transpose(img).convert("RGBA")
    return img

def image_to_np_rgb(img):
    # convert PIL RGBA to RGB numpy uint8
    if img.mode == "RGBA":
        # composite over white to avoid transparency surprises
        background = Image.new("RGBA", img.size, (255,255,255,255))
        img = Image.alpha_composite(background, img).convert("RGB")
    else:
        img = img.convert("RGB")
    return np.array(img, dtype=np.uint8)

def np_to_png_bytes(arr):
    img = Image.fromarray(arr.astype(np.uint8))
    buf = BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf

def check_cover_secret_sizes(cover_arr, secret_arr):
    Hc, Wc = cover_arr.shape[:2]
    Hs, Ws = secret_arr.shape[:2]
    capacity_bits = Hc * Wc * CHANNELS * BITS_PER_CHANNEL
    secret_bits = Hs * Ws * 3 * 8  # secret as 8-bit RGB
    return capacity_bits >= secret_bits

# --------- LSB encode/decode ---------
def lsb_embed(cover_arr, secret_arr):
    # cover_arr: HxWx3 uint8, secret_arr: hsxwsx3 uint8
    Hc, Wc = cover_arr.shape[:2]
    Hs, Ws = secret_arr.shape[:2]

    # flatten cover and secret channel-wise into bitstreams
    cover_flat = cover_arr.reshape(-1)  # bytes
    secret_flat = secret_arr.reshape(-1)

    # convert secret bytes to bits
    secret_bits = np.unpackbits(secret_flat)  # array of 0/1 length 8*N
    # pad secret_bits to fit into cover capacity
    capacity = cover_flat.size * BITS_PER_CHANNEL
    if secret_bits.size > capacity:
        raise ValueError("Secret too large for chosen cover image and BITS_PER_CHANNEL")

    # create a copy of cover bytes to modify
    out_bytes = np.array(cover_flat, copy=True)

    # embed bits sequentially into LSBs of cover bytes
    # we will use first capacity bits; remaining cover bytes untouched
    # operate per byte: clear lowest BITS_PER_CHANNEL and set from secret_bits
    # pack secret_bits into bytes to insert into successive cover bytes
    # build a mask array for insertion
    # We'll embed one bit per channel per pixel if BITS_PER_CHANNEL==1
    # secret_bits currently length S; iterate and write
    bit_idx = 0
    for i in range(out_bytes.size):
        if bit_idx >= secret_bits.size:
            break
        # clear LSBs
        out_bytes[i] = (out_bytes[i] & (~((1 << BITS_PER_CHANNEL) - 1)))
        # form value from next BITS_PER_CHANNEL bits (if available)
        val = 0
        for b in range(BITS_PER_CHANNEL):
            if bit_idx < secret_bits.size:
                val = (val << 1) | int(secret_bits[bit_idx])
                bit_idx += 1
            else:
                val = (val << 1)
        out_bytes[i] |= val

    # reshape back to image
    out_arr = out_bytes.reshape(cover_arr.shape)
    return out_arr

def lsb_extract(stego_arr, secret_shape):
    # stego_arr: HxWx3 uint8, secret_shape: (Hs, Ws, 3)
    Hc, Wc = stego_arr.shape[:2]
    Hs, Ws = secret_shape[:2]
    secret_size = Hs * Ws * 3
    # how many bits we will read
    total_bits_needed = secret_size * 8

    stego_flat = stego_arr.reshape(-1)
    bits = []
    for byte in stego_flat:
        # extract lowest BITS_PER_CHANNEL bits from this byte
        for b in reversed(range(BITS_PER_CHANNEL)):
            bits.append((byte >> b) & 1)
            if len(bits) >= total_bits_needed:
                break
        if len(bits) >= total_bits_needed:
            break

    bits = np.array(bits, dtype=np.uint8)
    # pack bits into bytes
    if bits.size < total_bits_needed:
        raise ValueError("Not enough bits in stego to recover secret")
    bits = bits.reshape(-1, 8)
    bytes_out = np.packbits(bits, axis=1)[:,0]
    secret_flat = bytes_out.flatten()
    secret_arr = secret_flat.reshape((Hs, Ws, 3)).astype(np.uint8)
    return secret_arr

# --------- Endpoints ---------
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

        # convert to RGB numpy
        cover_arr = image_to_np_rgb(cover_img)
        secret_arr = image_to_np_rgb(secret_img)

        # optionally resize secret to fit - here we allow embedding at full secret size if fits,
        # otherwise automatically scale secret down while preserving aspect ratio
        if not check_cover_secret_sizes(cover_arr, secret_arr):
            # compute max pixels allowed for secret
            cap_bits = cover_arr.size * BITS_PER_CHANNEL
            max_secret_bytes = cap_bits // 8
            max_pixels = max_secret_bytes // 3
            # scale secret so Hs*Ws <= max_pixels
            h, w = secret_arr.shape[:2]
            scale = (max_pixels / (h*w)) ** 0.5
            if scale <= 0:
                return jsonify({"error":"cover too small to embed any secret"}), 400
            new_w = max(1, int(w * scale))
            new_h = max(1, int(h * scale))
            secret_img = secret_img.resize((new_w, new_h), Image.LANCZOS)
            secret_arr = image_to_np_rgb(secret_img)

        # embed
        stego_arr = lsb_embed(cover_arr, secret_arr)

        buf = np_to_png_bytes(stego_arr)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="stego.png")
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

@app.route("/decode", methods=["POST"])
def decode_route():
    try:
        if "stego" not in request.files or "secret_shape" not in request.form:
            return jsonify({"error":"upload 'stego' and provide 'secret_shape' as 'HxW' in form"}), 400

        stego_img = pil_open_fix(request.files["stego"])
        stego_arr = image_to_np_rgb(stego_img)

        # parse secret shape from form, format "HxW" (e.g. 128x128)
        shape_str = request.form.get("secret_shape")
        try:
            parts = shape_str.lower().split('x')
            Hs = int(parts[0]); Ws = int(parts[1])
        except Exception:
            return jsonify({"error":"invalid secret_shape format; use HxW (e.g. 128x128)"}), 400

        secret_arr = lsb_extract(stego_arr, (Hs, Ws, 3))
        buf = np_to_png_bytes(secret_arr)
        return send_file(buf, mimetype="image/png", as_attachment=True, download_name="extracted.png")
    except Exception as e:
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT",8080)))
