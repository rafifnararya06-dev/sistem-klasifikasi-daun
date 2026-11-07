"""
=============================================================================
APP.PY - FLASK BACKEND SISTEM KLASIFIKASI CITRA DAUN (UTS COMPUTER VISION)
=============================================================================
Mematuhi secara ketat alur pemrosesan Computer Vision:
  1. Input Citra:
     - Menerima citra daun berwarna (JPG, PNG, JPEG) dari antarmuka web.
  2. Pre-processing:
     - Resize seragam (256x256 piksel).
     - Filter Gaussian Blur untuk menghaluskan dan mereduksi noise/watermark halus.
  3. Segmentasi Grayscale & Otsu Thresholding:
     - Konversi ke Grayscale.
     - Penyesuaian adaptif terhadap 3 kondisi background: Putih, Kotak-kotak, dan Hitam.
     - Operasi Morfologi (Opening & Closing) untuk menghasilkan Citra Biner solid
       dan mempertahankan bentuk asli daun tanpa cacat.
  4. Ekstraksi Fitur (10 Fitur Terstandar):
     - Bentuk: Area, Perimeter, Circularity dari citra biner.
     - Masking bitwise_and pada citra warna asli.
     - Warna: Rata-rata ruang warna HSV (H, S, V) pada piksel daun ber-masking.
     - Tekstur: GLCM Contrast, Energy, Homogeneity, Correlation.
  5. Klasifikasi K-NN (K-Nearest Neighbors):
     - Membandingkan bentuk dan fitur daun input dengan seluruh daun di folder 'Citra Daun'.
     - Menyesuaikan nama daun input dan menghitung persentase kemiripan daun yang mirip.
  6. Tingkat Akurasi Model (%):
     - Menampilkan persentase akurasi model teruji.
  7. Analisis Fitur:
     - Menampilkan kontribusi dan analisis fitur mana yang paling berpengaruh di web.
=============================================================================
"""

import os
import json
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

import feature_extraction as fe
import model_training as mt
import database as db

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CITRA_DIR = os.path.join(BASE_DIR, "Citra Daun")
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder=BASE_DIR, static_folder=BASE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Maksimal 32MB upload
app.config["TEMPLATES_AUTO_RELOAD"] = True
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

# Pastikan Browser Selalu Mengambil Versi Terbaru Tanpa Tertahan Cache
@app.after_request
def add_no_cache_headers(response):
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0, post-check=0, pre-check=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response

# Inisialisasi DB (opsional)
try:
    db.init_db()
except Exception as e:
    print(f"[DB INFO] Inisialisasi DB: {e}")


def make_json_safe(obj):
    """
    Mengonversi secara rekursif setiap tipe numpy array, float32, int32, dll.
    ke tipe native Python agar aman diserialisasi ke JSON tanpa memicu TypeError.
    """
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items() if not isinstance(v, np.ndarray)}
    elif isinstance(obj, (list, tuple)):
        return [make_json_safe(item) for item in obj if not isinstance(item, np.ndarray)]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.floating, np.float32, np.float64)):
        return float(obj)
    elif isinstance(obj, (np.integer, np.int32, np.int64, np.uint8)):
        return int(obj)
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    elif hasattr(obj, "item"):
        return obj.item()
    return obj


@app.route("/")
def index():
    """Tampilan antarmuka website utama."""
    return render_template("index.html")


@app.route("/style.css")
def serve_css():
    """Melayani file stylesheet CSS."""
    return send_from_directory(BASE_DIR, "style.css", mimetype="text/css")


@app.route("/Citra Daun/<path:filename>")
def serve_citra_daun(filename):
    """Melayani citra daun referensi dari folder 'Citra Daun'."""
    return send_from_directory(CITRA_DIR, filename)


# =============================================================================
# ROUTE UTAMA API: PROSES LENGKAP PENGOLAHAN CITRA & KLASIFIKASI K-NN
# =============================================================================
@app.route("/api/classify", methods=["POST"])
@app.route("/api/process", methods=["POST"])
def classify_leaf():
    """
    Menjalankan alur pemrosesan 7 poin:
      1. Input citra
      2. Pre-processing (resize seragam 256x256, Gaussian Blur denoising)
      3. Segmentasi Grayscale & Otsu thresholding adaptif (Putih, Kotak-kotak, Hitam)
      4. Ekstraksi fitur (Bentuk, Warna HSV, Tekstur GLCM)
      5. Klasifikasi K-NN: membandingkan dengan seluruh daun di folder 'Citra Daun',
         menyesuaikan nama daun input dan menghitung persentase kemiripannya
      6. Tingkat akurasi sistem (%)
      7. Analisis fitur paling berpengaruh
    """
    img_bgr = None
    source_name = "daun_input.jpg"
    user_leaf_name = request.form.get("leaf_name", "").strip()

    # 1. Terima gambar daun berwarna
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        source_name = file.filename
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({
            "success": False,
            "error": "Silakan unggah citra daun yang valid (format JPG, PNG, atau JPEG)!"
        }), 400

    # Jika nama input tidak diisi manual, gunakan nama file yang diunggah
    if not user_leaf_name:
        user_leaf_name = os.path.splitext(source_name)[0].strip()

    try:
        # ---------------------------------------------------------------------
        # TAHAP 1 s/d 4: PRE-PROCESSING, SEGMENTASI, MASKING & EKSTRAKSI FITUR
        # ---------------------------------------------------------------------
        pipeline_res = fe.process_leaf_pipeline(img_bgr)
        feature_vector = pipeline_res["feature_vector"]

        # ---------------------------------------------------------------------
        # TAHAP 5 s/d 7: KLASIFIKASI K-NN & PERBANDINGAN DENGAN CITRA DAUN
        # ---------------------------------------------------------------------
        knn_res = mt.compare_and_classify_leaf(feature_vector, user_input_name=user_leaf_name)

        response_payload = {
            "success": True,
            "source_name": source_name,
            "user_leaf_name": user_leaf_name,

            # TAHAPAN VISUAL SESUAI ALUR:
            "visual_sequence": {
                "citra_asli": pipeline_res["visual"]["citra_asli"],
                "citra_preprocessing": pipeline_res["visual"]["citra_preprocessing"],
                "citra_grayscale": pipeline_res["visual"]["citra_grayscale"],
                "citra_biner": pipeline_res["visual"]["citra_biner"],
                "citra_segmented_gray": pipeline_res["visual"]["citra_segmented_gray"],
                "citra_masked": pipeline_res["visual"]["citra_masked"],
                "citra_kontur": pipeline_res["visual"]["citra_kontur"],
                "citra_xray": pipeline_res["visual"]["citra_xray"],
                "citra_morfometri": pipeline_res["visual"]["citra_morfometri"]
            },

            # KONDISI BACKGROUND TERDETEKSI & NILAI OTSU
            "background_condition": pipeline_res["background_condition"],
            "otsu_threshold": pipeline_res["otsu_threshold"],

            # DATA NUMERIK RIIL HASIL EKSTRAKSI (10 FITUR)
            "features_extracted": pipeline_res["features"],
            "feature_vector": [float(x) for x in feature_vector],

            # HASIL KLASIFIKASI K-NN & KEMIRIPAN TERHADAP CITRA DAUN
            "classification": {
                "predicted_leaf": knn_res["predicted_leaf"],
                "predicted_filename": knn_res["predicted_filename"],
                "scientific_name": knn_res["scientific_name"],
                "family": knn_res["family"],
                "bentuk_morfologi": knn_res["bentuk_morfologi"],
                "top_similarity_percentage": knn_res["top_similarity_percentage"],
                "confidence_percentage": knn_res["confidence_percentage"],
                "model_accuracy_percentage": knn_res["model_accuracy_percentage"],
                "dynamic_probabilities": knn_res["dynamic_probabilities"],
                "top_matches": knn_res["top_matches"],
                "all_similarities": knn_res["all_similarities"]
            },

            # ANALISIS FITUR PALING BERPENGARUH
            "feature_importance_analysis": {
                "ranking": knn_res["feature_ranking"],
                "kesimpulan": knn_res["kesimpulan_analisis"]
            }
        }

        return jsonify(make_json_safe(response_payload))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# ROUTE API: DATASET FITUR 30 DAUN LENGKAP
# =============================================================================
@app.route("/api/dataset", methods=["GET"])
def get_dataset_features():
    """
    Mengembalikan seluruh rekaman dataset fitur 30 citra daun dari 'dataset_features.json'.
    Menjawab permintaan: 'Jangan lupakan dataset fiturnya juga'.
    """
    dataset_json_path = os.path.join(BASE_DIR, "dataset_features.json")
    if not os.path.exists(dataset_json_path):
        mt.train_and_evaluate_model()

    try:
        with open(dataset_json_path, "r", encoding="utf-8") as f:
            records = json.load(f)
        return jsonify({
            "success": True,
            "total_records": len(records),
            "data": records
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# =============================================================================
# ROUTE API: METADATA MODEL & SAMPEL PENGUJIAN
# =============================================================================
@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Mengembalikan metadata model KNN, akurasi, dan daftar ranking fitur."""
    meta_path = os.path.join(BASE_DIR, "model_metadata.json")
    if not os.path.exists(meta_path):
        mt.train_and_evaluate_model()

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return jsonify({"success": True, "data": meta})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/sample/<path:leaf_key>", methods=["GET"])
def get_sample_leaf(leaf_key):
    """
    Menyediakan citra sampel daun dari folder 'Citra Daun' untuk pengujian instan.
    Mendukung nama file langsung atau kata kunci nama daun.
    """
    all_files = os.listdir(CITRA_DIR)
    target_file = None

    key_clean = leaf_key.lower().strip()
    for f in all_files:
        if f.lower() == key_clean or os.path.splitext(f)[0].lower().strip() == key_clean:
            target_file = f
            break

    if target_file is None:
        # Coba pencarian substring
        for f in all_files:
            if key_clean in f.lower():
                target_file = f
                break

    if target_file is None:
        return jsonify({"success": False, "error": f"Citra '{leaf_key}' tidak ditemukan di folder Citra Daun."}), 404

    img_path = os.path.join(CITRA_DIR, target_file)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return jsonify({"success": False, "error": f"Gagal membaca citra '{target_file}'."}), 500

    return jsonify({
        "success": True,
        "file_name": target_file,
        "leaf_name": os.path.splitext(target_file)[0].strip(),
        "image_base64": fe.image_to_base64(img_bgr, is_bgr=True)
    })


# Server tidak dijalankan otomatis sesuai instruksi pengguna
if __name__ == "__main__":
    print("=" * 65)
    print("SISTEM KLASIFIKASI CITRA DAUN (UTS COMPUTER VISION)")
    print("Server siap dijalankan secara manual oleh pengguna.")
    print("Buka browser di: http://localhost:5000")
    print("=" * 65)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
