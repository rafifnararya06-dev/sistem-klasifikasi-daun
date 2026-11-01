"""
=============================================================================
APP.PY - FLASK BACKEND KLASIFIKASI & ANALISIS CITRA DAUN (UTS CV REVISI)
=============================================================================
Alur Processing yang Benar:
  1. Input Citra (Pilih citra daun dari File Explorer)
  2. Segmentasi Pemisahan Background (Isolasi daun dari latar belakang)
  3. Gambar Berwarna ke Grayscale (Konversi RGB/BGR ke tingkat keabuan)
  4. Grayscale ke Thresholding dengan 3 Metode Berbeda (Global, Adaptive, Otsu)
  5. Ekstraksi dari Warna, Tekstur, dan Bentuk (RGB/HSV, GLCM/Venasi, Morfometri)
  6. Klasifikasi & Perbandingan Kemiripan dengan Jenis Daun Lain
     - Menentukan daun apa yang paling mirip
     - Menghitung persentase kemiripan komprehensif
     - Menyajikan tabel komparasi peringkat seluruh daun referensi
=============================================================================
"""

import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, send_from_directory

import leaf_classifier as lc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

app = Flask(__name__, template_folder=BASE_DIR, static_folder=BASE_DIR)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Maksimal 16MB


def make_json_safe(obj):
    """Mengonversi secara rekursif setiap tipe numpy/ndarray ke tipe Python agar aman diserialisasi ke JSON."""
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
    """Tampilan utama web sistem pengolahan citra daun."""
    return render_template("index.html")


@app.route("/style.css")
def serve_css():
    """Melayani file CSS."""
    return send_from_directory(BASE_DIR, "style.css", mimetype="text/css")


# =============================================================================
# ROUTE API: PROSES PIPELINE LENGKAP (LANGKAH 1 s/d 6)
# =============================================================================
@app.route("/api/process", methods=["POST"])
def process_pipeline():
    """
    Menerima citra yang dipilih dari File Explorer (Upload) dan metode thresholding.
    Menjalankan alur pemrosesan:
      1. Input Citra
      2. Segmentasi Pemisahan Background
      3. Gambar Berwarna ke Grayscale
      4. Grayscale ke Thresholding dengan 3 Metode Berbeda (Global, Adaptive, Otsu)
      5. Ekstraksi dari Warna, Tekstur, dan Bentuk
      6. Klasifikasi & Perbandingan Kemiripan dengan Jenis Daun Lain
    """
    img_bgr = None
    source_name = "Citra Daun"
    leaf_name = request.form.get("leaf_name", "").strip()
    metode_threshold = request.form.get("threshold_method", "otsu")

    # 1. Mengambil file dari File Explorer (Upload)
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        source_name = file.filename
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    # Validasi input
    if img_bgr is None:
        return jsonify({
            "success": False,
            "error": "Silakan pilih citra daun dari File Explorer terlebih dahulu!"
        }), 400

    try:
        # Ekstrak atau pastikan nama daun dari citra yang dipilih
        if not leaf_name:
            leaf_name = lc.extract_leaf_name(source_name)

        # Jalankan 6 Langkah Pemrosesan Lengkap
        pipeline_res = lc.proses_citra_lengkap(
            img_bgr,
            metode_threshold=metode_threshold,
            leaf_name=leaf_name,
            source_filename=source_name
        )

        # Hasilkan 10 variasi citra visual
        variations_raw = lc.generate_10_variations(img_bgr)
        output_10_citra = []
        for var in variations_raw:
            output_10_citra.append({
                "name": var["name"],
                "base64": lc.image_to_base64(var["img"], is_bgr=True)
            })

        # Bentuk dataset fitur 10 baris dari citra yang dipilih
        dataset_10 = lc.buat_dataset_10_fitur(
            img_bgr,
            nama_daun=leaf_name,
            metode_threshold=metode_threshold
        )

        response_data = {
            "success": True,
            "source_name": source_name,
            "leaf_name": leaf_name,
            "threshold_method_chosen": metode_threshold,

            # 6 LANGKAH PROCESSING
            "step1_input": pipeline_res["step1_input"],
            "step2_segmentasi": pipeline_res["step2_segmentasi"],
            "step3_grayscale": pipeline_res["step3_grayscale"],
            "step4_thresholding": pipeline_res["step4_thresholding"],
            "step5_ekstraksi": pipeline_res["step5_ekstraksi"],
            "step6_klasifikasi": pipeline_res["step6_klasifikasi"],

            # Visual 10 Citra & Dataset Fitur
            "output_10_citra": output_10_citra,
            "dataset_10": {
                "feature_names": lc.FEATURE_NAMES,
                "records": dataset_10
            }
        }

        safe_response = make_json_safe(response_data)
        return jsonify(safe_response)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# Server tidak dijalankan otomatis sesuai instruksi user
if __name__ == "__main__":
    print("=" * 60)
    print("SISTEM PENGOLAHAN CITRA DAUN - UTS CV")
    print("Server siap dijalankan via batch file oleh pengguna.")
    print("Buka browser di: http://localhost:5000")
    print("=" * 60)
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
