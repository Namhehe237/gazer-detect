from flask import Flask, request, jsonify
from deepface import DeepFace
from dotenv import load_dotenv
import os
from PIL import Image
import numpy as np
import cv2

load_dotenv()

app = Flask(__name__)
MODEL_NAME = os.getenv("MODEL_NAME", "Facenet")  # Facenet nhanh hơn ArcFace
DETECTOR = os.getenv("DETECTOR", "opencv")  # opencv nhanh nhất

CUSTOM_THRESHOLD = {
    "Facenet": 0.40,
    "Facenet512": 0.30,
    "ArcFace": 0.75,
    "VGG-Face": 0.40,
}

def read_image(file_storage, max_size=640):
    """Đọc và resize ảnh để tăng tốc"""
    try:
        img = Image.open(file_storage.stream).convert("RGB")
        
        # Resize ảnh nếu quá lớn (giảm 30-50% thời gian)
        if max(img.size) > max_size:
            img.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        
        return np.array(img)
    except Exception as e:
        raise ValueError(f"Lỗi đọc ảnh: {str(e)}")

@app.route("/compare-faces", methods=["POST"])
def compare_faces():
    try:
        img1_file = request.files.get("image1")
        img2_file = request.files.get("image2")

        if not img1_file or not img2_file:
            return jsonify({"error": "Thiếu file ảnh"}), 400

        # Đọc và resize ảnh
        img1 = read_image(img1_file, max_size=640)
        img2 = read_image(img2_file, max_size=640)

        # Cấu hình tối ưu tốc độ
        result = DeepFace.verify(
            img1_path=img1,
            img2_path=img2,
            model_name=MODEL_NAME,
            detector_backend=DETECTOR,  # opencv nhanh nhất
            enforce_detection=True,
            align=True,
            normalization="base"  # Nhanh hơn "ArcFace"
        )

        custom_threshold = CUSTOM_THRESHOLD.get(MODEL_NAME, result["threshold"])
        custom_verified = result["distance"] <= custom_threshold
        
        return jsonify({
            "same_person": custom_verified,
            "distance": float(result["distance"]),
            "threshold": float(custom_threshold),
            "confidence": max(0, 1 - (result["distance"] / custom_threshold)),
            "model": MODEL_NAME
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/", methods=["GET"])
def index():
    return jsonify({
        "message": "Fast Face Comparison API 🚀",
        "model": MODEL_NAME,
        "detector": DETECTOR
    })


if __name__ == "__main__":
    app.run(
        host=os.getenv("FLASK_RUN_HOST", "0.0.0.0"),
        port=int(os.getenv("FLASK_RUN_PORT", 5000)),
        debug=(os.getenv("FLASK_ENV") == "development")
    )