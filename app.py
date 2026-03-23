"""
Flower Species & Color Detection Web App
=========================================
Flask backend that loads the trained EfficientNetB0 model,
accepts uploaded images, runs GrabCut segmentation + K-Means
color extraction, and returns JSON results.
"""

import os
import io
import base64
import numpy as np
import cv2
from collections import Counter
from sklearn.cluster import KMeans

import tensorflow as tf
from flask import Flask, request, jsonify, render_template

# Configuration
IMG_SIZE = (224, 224)
MODEL_PATH = "flower_species_and_color_model.h5"
CLASS_NAMES = ["daisy", "dandelion", "rose", "sunflower", "tulip"]

# Load model once at startup
print("[*] Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)
print("[OK] Model loaded successfully.")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB upload limit


def segment_flower(img_bgr):
    """Cut out the flower from background using OpenCV GrabCut."""
    orig_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    mask = np.zeros(img_bgr.shape[:2], np.uint8)
    bgd = np.zeros((1, 65), np.float64)
    fgd = np.zeros((1, 65), np.float64)
    h, w = img_bgr.shape[:2]
    rect = (int(w * 0.1), int(h * 0.1), int(w * 0.8), int(h * 0.8))
    cv2.grabCut(img_bgr, mask, rect, bgd, fgd, 5, cv2.GC_INIT_WITH_RECT)
    mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype("uint8")
    # White background instead of black for the cutout
    cutout = np.full_like(orig_rgb, 255)
    cutout[mask2 == 1] = orig_rgb[mask2 == 1]
    return orig_rgb, cutout, mask2


def get_accurate_colors(image_rgb, mask, n_colors=3):
    """Return the top dominant RGB colors of the flower foreground."""
    pixels = image_rgb.reshape(-1, 3)
    fg = pixels[mask.reshape(-1) == 1]
    if len(fg) == 0:
        return [(0, 0, 0)]
    k = min(n_colors, len(fg))
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(fg)
    counts = Counter(km.labels_)
    ordered = counts.most_common()
    colors = []
    for cluster_id, _ in ordered:
        c = tuple(int(v) for v in km.cluster_centers_[cluster_id])
        colors.append(c)
    return colors


_CSS_COLORS = {
    # Reds
    "Dark Red": (139, 0, 0), "Red": (255, 0, 0), "Light Red": (255, 102, 102),
    # Oranges
    "Dark Orange": (255, 140, 0), "Orange": (255, 165, 0), "Light Orange": (255, 204, 153),
    # Yellows
    "Dark Yellow": (204, 204, 0), "Yellow": (255, 255, 0), "Light Yellow": (255, 255, 153),
    # Greens
    "Dark Green": (0, 100, 0), "Green": (0, 128, 0), "Light Green": (144, 238, 144),
    # Blues
    "Dark Blue": (0, 0, 139), "Blue": (0, 0, 255), "Light Blue": (173, 216, 230),
    # Purples
    "Dark Purple": (75, 0, 130), "Purple": (128, 0, 128), "Light Purple": (216, 191, 216),
    # Pinks
    "Dark Pink": (255, 20, 147), "Pink": (255, 192, 203), "Light Pink": (255, 182, 193),
    # Browns
    "Dark Brown": (101, 67, 33), "Brown": (139, 69, 19), "Light Brown": (210, 180, 140),
    # Neutrals
    "White": (255, 255, 255),
    "Light Gray": (211, 211, 211), "Gray": (128, 128, 128), "Dark Gray": (64, 64, 64),
    "Black": (0, 0, 0)
}


def rgb_to_name(rgb):
    """Map an RGB tuple to the closest CSS color name."""
    min_dist = float("inf")
    name = "Unknown"
    for n, c in _CSS_COLORS.items():
        d = sum((a - b) ** 2 for a, b in zip(rgb, c))
        if d < min_dist:
            min_dist = d
            name = n
    return name


def img_to_b64(img_rgb):
    """Convert an RGB numpy array to a base64 JPEG data URI."""
    bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return "data:image/jpeg;base64," + base64.b64encode(buf).decode()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    # Read into memory
    file_bytes = file.read()
    np_arr = np.frombuffer(file_bytes, np.uint8)
    img_bgr = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    if img_bgr is None:
        return jsonify({"error": "Invalid image"}), 400

    # 1. Species prediction
    img_resized = cv2.resize(img_bgr, IMG_SIZE)
    img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB).astype("float32")
    img_batch = np.expand_dims(img_rgb, axis=0)
    preds = model.predict(img_batch, verbose=0)
    confidence = float(np.max(preds) * 100)
    species = CLASS_NAMES[int(np.argmax(preds))]

    # 2. Segmentation & color
    orig_rgb, cutout, mask = segment_flower(img_bgr)
    colors = get_accurate_colors(orig_rgb, mask, n_colors=3)
    dominant = colors[0]
    color_name = rgb_to_name(dominant)

    # 3. Encode images for frontend
    orig_b64 = img_to_b64(orig_rgb)
    cutout_b64 = img_to_b64(cutout)

    return jsonify({
        "species": species,
        "confidence": round(confidence, 2),
        "dominant_color": {
            "rgb": list(dominant),
            "hex": "#{:02x}{:02x}{:02x}".format(*dominant),
            "name": color_name,
        },
        "original_image": orig_b64,
        "cutout_image": cutout_b64,
    })


if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=5000)
