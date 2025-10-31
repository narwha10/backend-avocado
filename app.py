import os
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, jsonify
from flask_cors import CORS
from joblib import load

# ============================================
# 🥑 AVOCADO RIPENESS CLASSIFIER – FLASK BACKEND
# ============================================
app = Flask(__name__)
CORS(app)

# === Path model & extractor ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "model", "model_alpukat.pkl")
FEATURE_EXTRACTOR_PATH = os.path.join(BASE_DIR, "model", "feature_extractor.pkl")


# ============================================
# 🎨 Class Feature Extractor (wajib ada di namespace)
# ============================================
class ColorHistogramExtractor:
    def __init__(self, img_size=128, bins=(8, 8, 8)):
        self.img_size = img_size
        self.bins = bins

    def extract(self, image):
        """Hitung histogram warna untuk satu gambar."""
        if image is None:
            raise ValueError("Gambar tidak valid (None).")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (self.img_size, self.img_size))

        hist = cv2.calcHist(
            [image],
            [0, 1, 2],
            None,
            self.bins,
            [0, 256, 0, 256, 0, 256],
        )
        hist = cv2.normalize(hist, hist).flatten()
        return hist

    def transform(self, image_array):
        """
        Mendukung input array dari Flask (numpy RGB).
        Kalau input dari preprocessing flatten, langsung kembalikan.
        """
        if isinstance(image_array, np.ndarray):
            if image_array.ndim == 4:  # batch
                return np.array([self.extract(img) for img in image_array])
            elif image_array.ndim == 3:  # single image
                return np.array([self.extract(image_array)])
            elif image_array.ndim == 2:  # sudah flatten
                return image_array
        raise ValueError("Format array gambar tidak dikenali.")


# ============================================
# 📦 Load model & feature extractor
# ============================================
model = None
try:
    model = load(MODEL_PATH)
    print(f"✅ Model berhasil dimuat dari: {MODEL_PATH}")
except Exception as e:
    print(f"❌ Gagal memuat model utama: {e}")

feature_extractor = None
try:
    if os.path.exists(FEATURE_EXTRACTOR_PATH):
        feature_extractor = load(FEATURE_EXTRACTOR_PATH)
        print(f"✅ Feature extractor dimuat dari: {FEATURE_EXTRACTOR_PATH}")
    else:
        print("⚠️ Tidak ada file feature_extractor.pkl, akan menggunakan flatten default.")
except Exception as e:
    print(f"⚠️ Gagal memuat feature extractor: {e}")

# === Label klasifikasi ===
LABELS = ["Belum Matang", "Matang"]


# ============================================
# 🔍 Fungsi bantu – Preprocessing gambar
# ============================================
def preprocess_image(image_bytes):
    """Konversi gambar menjadi vektor fitur sesuai feature extractor."""
    try:
        image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        raise ValueError(f"Gagal membuka gambar: {e}")

    # Konversi ke array uint8
    image_array = np.array(image, dtype=np.uint8)

    # Gunakan extractor jika ada
    if feature_extractor is not None:
        try:
            # Jika extractor punya .extract() → panggil itu
            if hasattr(feature_extractor, "extract"):
                features = feature_extractor.extract(image_array)
                features = np.array(features).reshape(1, -1)
            else:
                # Jika hanya punya .transform() → kemungkinan itu PCA/Scaler
                features = feature_extractor.transform([image_array.flatten()])
        except Exception as e:
            print(f"⚠️ Gagal pakai feature_extractor, fallback ke flatten: {e}")
            features = image_array.flatten().reshape(1, -1)
    else:
        # Jika tidak ada extractor → pakai flatten default
        features = image_array.flatten().reshape(1, -1)

    return features

# ============================================
# 🌐 ROUTES
# ============================================
@app.route("/")
def home():
    return jsonify({
        "message": "🍈 Avocado Ripeness Recognition API is running!",
        "status": "OK"
    })


@app.route("/predict", methods=["POST"])
def predict():
    if model is None:
        return jsonify({"error": "Model belum dimuat di server"}), 500

    if "file" not in request.files:
        return jsonify({"error": "Tidak ada file yang diunggah (field 'file' kosong)"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Nama file kosong"}), 400

    try:
        image_bytes = file.read()
        if not image_bytes:
            return jsonify({"error": "File kosong"}), 400

        # === Ekstraksi fitur ===
        features = preprocess_image(image_bytes)

        # === Validasi dimensi ===
        if hasattr(model, "n_features_in_") and features.shape[1] != model.n_features_in_:
            return jsonify({
                "error": "Dimensi fitur tidak cocok",
                "expected": int(model.n_features_in_),
                "received": int(features.shape[1])
            }), 400

        # === Prediksi ===
        prediction = model.predict(features)[0]
        confidence = 1.0

        if hasattr(model, "predict_proba"):
            confidence = float(np.max(model.predict_proba(features)))

        label = LABELS[prediction] if isinstance(prediction, (int, np.integer)) and prediction < len(LABELS) else str(prediction)

        return jsonify({
            "label": label,
            "confidence": round(confidence, 4)
        })

    except Exception as e:
        print(f"❌ Error saat prediksi: {e}")
        return jsonify({"error": "Prediction failed", "details": str(e)}), 500


# ============================================
# 🚀 MAIN
# ============================================
if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
