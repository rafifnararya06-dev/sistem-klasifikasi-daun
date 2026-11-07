"""
=============================================================================
API/INDEX.PY - FLASK BACKEND SERVERLESS UNTUK VERCEL (UI BARU)
=============================================================================
Backend serverless untuk melayani UI Baru sistem klasifikasi citra daun di Vercel:
  1. Input Citra: Menerima upload citra atau sampel uji cepat
  2. Pre-processing: Resize 256x256, Gaussian Blur denoising
  3. Segmentasi: Grayscale & Otsu adaptif (Putih, Kotak-kotak, Hitam) + Morfologi
  4. Ekstraksi Fitur: 10 Dimensi (Bentuk, Warna HSV, Tekstur GLCM)
  5. Klasifikasi K-NN: Menyesuaikan nama daun input & menghitung kemiripan
  6. Akurasi Model & Dataset Fitur 30 Daun
  7. Analisis Fitur Paling Berpengaruh (Dinamis)
=============================================================================
"""

import os
import sys
import json
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory

# ---------------------------------------------------------------------------
# Path Setup: pastikan modul feature_extraction dan model_training bisa di-import
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import feature_extraction as fe
import model_training as mt

PUBLIC_DIR = os.path.join(ROOT_DIR, "public")
CITRA_DIR = os.path.join(ROOT_DIR, "Citra Daun")
PUBLIC_CITRA_DIR = os.path.join(PUBLIC_DIR, "Citra Daun")
DATASET_JSON_PATH = os.path.join(ROOT_DIR, "dataset_features.json")
METADATA_PATH = os.path.join(ROOT_DIR, "model_metadata.json")

# Inisialisasi Flask
app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024  # Maks 32MB upload
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0


@app.after_request
def add_no_cache_headers(response):
    """Mencegah caching browser agar perubahan UI selalu tampil segar."""
    response.headers["Cache-Control"] = "no-cache, no-store, must-revalidate, max-age=0"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "-1"
    return response


def make_json_safe(obj):
    """Konversi rekursif tipe data numpy ke native Python untuk JSON serialization."""
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


# ---------------------------------------------------------------------------
# STATIC ROUTES (Fallback jika tidak tertangani oleh Vercel Static Layer)
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Halaman utama antarmuka baru."""
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/style.css")
def serve_css():
    """Stylesheet antarmuka baru."""
    return send_from_directory(PUBLIC_DIR, "style.css", mimetype="text/css")


@app.route("/Citra Daun/<path:filename>")
def serve_citra_daun(filename):
    """Melayani gambar daun referensi dari public/Citra Daun atau Citra Daun."""
    if os.path.exists(os.path.join(PUBLIC_CITRA_DIR, filename)):
        return send_from_directory(PUBLIC_CITRA_DIR, filename)
    elif os.path.exists(os.path.join(CITRA_DIR, filename)):
        return send_from_directory(CITRA_DIR, filename)
    return jsonify({"error": f"Citra '{filename}' tidak ditemukan"}), 404


# ---------------------------------------------------------------------------
# API HEALTH STATUS
# ---------------------------------------------------------------------------
@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "online",
        "platform": "vercel-serverless",
        "dataset_loaded": os.path.exists(DATASET_JSON_PATH),
        "metadata_loaded": os.path.exists(METADATA_PATH)
    })


# ---------------------------------------------------------------------------
# ROUTE UTAMA API: PROSES PENGOLAHAN CITRA & KLASIFIKASI K-NN
# ---------------------------------------------------------------------------
@app.route("/api/classify", methods=["POST"])
@app.route("/api/process", methods=["POST"])
def classify_leaf():
    """
    Menjalankan alur pemrosesan 7 poin untuk UI Baru:
      1. Input citra
      2. Pre-processing (resize seragam 256x256, Gaussian Blur denoising)
      3. Segmentasi Grayscale & Otsu adaptif
      4. Ekstraksi fitur (Bentuk, Warna HSV, Tekstur GLCM)
      5. Klasifikasi K-NN: menyesuaikan nama daun input & menghitung kemiripan
      6. Tingkat akurasi sistem (%)
      7. Analisis fitur paling berpengaruh (Dinamis)
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

    if not user_leaf_name:
        user_leaf_name = os.path.splitext(source_name)[0].strip()

    try:
        # TAHAP 1 s/d 4: PRE-PROCESSING, SEGMENTASI & EKSTRAKSI FITUR
        pipeline_res = fe.process_leaf_pipeline(img_bgr)
        feature_vector = pipeline_res["feature_vector"]

        # TAHAP 5 s/d 7: KLASIFIKASI K-NN & PERBANDINGAN DENGAN CITRA DAUN
        knn_res = mt.compare_and_classify_leaf(feature_vector, user_input_name=user_leaf_name)

        response_payload = {
            "success": True,
            "source_name": source_name,
            "user_leaf_name": user_leaf_name,

            # TAHAPAN VISUAL
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

            # KONDISI BACKGROUND & OTSU THRESHOLD
            "background_condition": pipeline_res["background_condition"],
            "otsu_threshold": pipeline_res["otsu_threshold"],

            # 10 FITUR EKSTRAKSI
            "features_extracted": pipeline_res["features"],
            "feature_vector": [float(x) for x in feature_vector],

            # HASIL KLASIFIKASI K-NN & KEMIRIPAN
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


# ---------------------------------------------------------------------------
# ROUTE API: DATASET FITUR 30 CITRA DAUN
# ---------------------------------------------------------------------------
@app.route("/api/dataset", methods=["GET"])
def get_dataset_features():
    """Mengembalikan data fitur 30 citra daun dari dataset_features.json."""
    if not os.path.exists(DATASET_JSON_PATH):
        try:
            mt.train_and_evaluate_model()
        except Exception:
            pass

    try:
        with open(DATASET_JSON_PATH, "r", encoding="utf-8") as f:
            records = json.load(f)
        return jsonify({
            "success": True,
            "total_records": len(records),
            "data": records
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# ROUTE API: METADATA MODEL & AKURASI
# ---------------------------------------------------------------------------
@app.route("/api/metadata", methods=["GET"])
def get_metadata():
    """Mengembalikan metadata model KNN, akurasi, dan bobot fitur."""
    if not os.path.exists(METADATA_PATH):
        try:
            mt.train_and_evaluate_model()
        except Exception:
            pass

    try:
        with open(METADATA_PATH, "r", encoding="utf-8") as f:
            meta = json.load(f)
        return jsonify({"success": True, "data": meta})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# ROUTE API: SAMPEL PENGUJIAN INSTAN
# ---------------------------------------------------------------------------
@app.route("/api/sample/<path:leaf_key>", methods=["GET"])
def get_sample_leaf(leaf_key):
    """Menyediakan citra sampel daun dari folder 'Citra Daun' atau 'public/Citra Daun'."""
    search_dirs = [CITRA_DIR, PUBLIC_CITRA_DIR]
    all_files = []
    found_dir = None

    for d in search_dirs:
        if os.path.exists(d):
            all_files = os.listdir(d)
            found_dir = d
            break

    target_file = None
    key_clean = leaf_key.lower().strip()

    for f in all_files:
        if f.lower() == key_clean or os.path.splitext(f)[0].lower().strip() == key_clean:
            target_file = f
            break

    if target_file is None:
        for f in all_files:
            if key_clean in f.lower():
                target_file = f
                break

    if target_file is None or found_dir is None:
        return jsonify({"success": False, "error": f"Citra '{leaf_key}' tidak ditemukan di koleksi sampel."}), 404

    img_path = os.path.join(found_dir, target_file)
    img_bgr = cv2.imread(img_path)
    if img_bgr is None:
        return jsonify({"success": False, "error": f"Gagal membaca citra '{target_file}'."}), 500

    return jsonify({
        "success": True,
        "file_name": target_file,
        "leaf_name": os.path.splitext(target_file)[0].strip(),
        "image_base64": fe.image_to_base64(img_bgr, is_bgr=True)
    })


# Export untuk Vercel Serverless
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
