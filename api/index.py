"""
=============================================================================
API/INDEX.PY - FLASK BACKEND UNTUK VERCEL SERVERLESS
=============================================================================
Adaptasi dari app.py agar kompatibel dengan Vercel Serverless Functions.

Perubahan utama dari app.py lokal:
  1. SQLite diganti dengan riwayat in-memory (list Python) karena Vercel
     tidak mendukung filesystem persistensi di antara invokasi.
  2. Path file referensi daun disesuaikan relatif terhadap lokasi api/index.py.
  3. Flask app di-export via variable `app` agar Vercel bisa mendeteksinya.
=============================================================================
"""

import os
import sys
import cv2
import numpy as np
from flask import Flask, request, jsonify, send_from_directory
from datetime import datetime

# ---------------------------------------------------------------------------
# Path Setup: pastikan modul leaf_classifier bisa di-import dari root project
# ---------------------------------------------------------------------------
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import leaf_classifier as lc

# ---------------------------------------------------------------------------
# Inisialisasi Flask
# ---------------------------------------------------------------------------
PUBLIC_DIR = os.path.join(ROOT_DIR, "public")

app = Flask(__name__, static_folder=PUBLIC_DIR, static_url_path="")
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Maks 16 MB per upload

# ---------------------------------------------------------------------------
# Penyimpanan Riwayat In-Memory (pengganti SQLite untuk Vercel)
# Catatan: data ini bersifat sementara dan direset saat cold start serverless.
# ---------------------------------------------------------------------------
_riwayat_store: list = []
_riwayat_counter: int = 0


def _simpan_riwayat_memory(pipeline_res, source_name, leaf_name, metode_threshold, dataset_10):
    """Menyimpan hasil klasifikasi ke list in-memory dan mengembalikan ID."""
    global _riwayat_counter, _riwayat_store
    _riwayat_counter += 1
    riwayat_id = _riwayat_counter

    step1   = pipeline_res.get("step1_input", {})
    step5   = pipeline_res.get("step5_ekstraksi", {})
    step6   = pipeline_res.get("step6_klasifikasi", {})
    warna   = step5.get("warna", {})
    bentuk  = step5.get("bentuk", {})
    tekstur = step5.get("tekstur", {})
    daftar  = step6.get("daftar_perbandingan", [])
    top     = daftar[0] if daftar else {}

    record = {
        "id":                   riwayat_id,
        "waktu":                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "nama_file":            source_name,
        "nama_daun_input":      leaf_name,
        "metode_threshold":     metode_threshold,
        "daun_teridentifikasi": step6.get("predicted_class"),
        "nama_ilmiah":          step6.get("scientific_name"),
        "famili":               step6.get("family"),
        "persentase_total":     top.get("persentase_total"),
        "persentase_warna":     top.get("persentase_warna"),
        "persentase_bentuk":    top.get("persentase_bentuk"),
        "persentase_tekstur":   top.get("persentase_tekstur"),
        "status_kemiripan":     top.get("status_label"),
        "R_mean":               warna.get("R_mean"),
        "G_mean":               warna.get("G_mean"),
        "B_mean":               warna.get("B_mean"),
        "R_std":                warna.get("R_std"),
        "G_std":                warna.get("G_std"),
        "B_std":                warna.get("B_std"),
        "H_mean":               warna.get("H_mean"),
        "S_mean":               warna.get("S_mean"),
        "V_mean":               warna.get("V_mean"),
        "hex_warna":            warna.get("hex"),
        "kategori_warna":       warna.get("kategori_warna"),
        "area_px":              bentuk.get("main_area"),
        "perimeter_px":         bentuk.get("main_perimeter"),
        "metric":               bentuk.get("metric"),
        "eccentricity":         bentuk.get("eccentricity"),
        "aspect_ratio":         bentuk.get("aspect_ratio"),
        "solidity":             bentuk.get("solidity"),
        "extent":               bentuk.get("extent"),
        "jumlah_objek":         bentuk.get("jumlah_objek"),
        "kontras":              tekstur.get("kontras"),
        "dissimilarity":        tekstur.get("dissimilarity"),
        "homogenitas":          tekstur.get("homogenitas"),
        "energi":               tekstur.get("energi"),
        "korelasi":             tekstur.get("korelasi"),
        "venation_density":     tekstur.get("venation_density"),
        "dimensi_asli":         step1.get("orig_dim"),
        "dimensi_standar":      step1.get("dim_standard"),
        "dataset_fitur":        dataset_10,
    }
    _riwayat_store.insert(0, record)
    # Batasi agar list tidak tumbuh tak terbatas dalam satu sesi serverless
    if len(_riwayat_store) > 200:
        _riwayat_store = _riwayat_store[:200]
    return riwayat_id


# ---------------------------------------------------------------------------
# Helper: Konversi tipe numpy agar aman di JSON
# ---------------------------------------------------------------------------
def make_json_safe(obj):
    """Mengonversi secara rekursif tipe numpy/ndarray ke tipe Python standar."""
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
# ROUTE: Halaman Utama
# ---------------------------------------------------------------------------
@app.route("/")
def index():
    """Sajikan index.html dari folder public."""
    return send_from_directory(PUBLIC_DIR, "index.html")


@app.route("/style.css")
def serve_css():
    """Sajikan style.css dari folder public."""
    return send_from_directory(PUBLIC_DIR, "style.css", mimetype="text/css")


# ---------------------------------------------------------------------------
# ROUTE API: PROSES PIPELINE LENGKAP (LANGKAH 1 s/d 6)
# ---------------------------------------------------------------------------
@app.route("/api/process", methods=["POST"])
def process_pipeline():
    """
    Menerima citra upload dari File Explorer dan metode thresholding.
    Menjalankan 6 langkah pemrosesan citra daun secara lengkap.
    """
    img_bgr = None
    source_name = "Citra Daun"
    leaf_name = request.form.get("leaf_name", "").strip()
    metode_threshold = request.form.get("threshold_method", "otsu")

    # Ambil file dari upload
    if "file" in request.files and request.files["file"].filename != "":
        file = request.files["file"]
        source_name = file.filename
        file_bytes = np.frombuffer(file.read(), np.uint8)
        img_bgr = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)

    if img_bgr is None:
        return jsonify({
            "success": False,
            "error": "Silakan pilih citra daun dari File Explorer terlebih dahulu!"
        }), 400

    try:
        if not leaf_name:
            leaf_name = lc.extract_leaf_name(source_name)

        # Jalankan 6 langkah pipeline
        pipeline_res = lc.proses_citra_lengkap(
            img_bgr,
            metode_threshold=metode_threshold,
            leaf_name=leaf_name,
            source_filename=source_name
        )

        # Generate 10 variasi citra
        variations_raw = lc.generate_10_variations(img_bgr)
        output_10_citra = [
            {
                "name": var["name"],
                "base64": lc.image_to_base64(var["img"], is_bgr=True)
            }
            for var in variations_raw
        ]

        # Dataset fitur 10 baris
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
            "step1_input":       pipeline_res["step1_input"],
            "step2_segmentasi":  pipeline_res["step2_segmentasi"],
            "step3_grayscale":   pipeline_res["step3_grayscale"],
            "step4_thresholding": pipeline_res["step4_thresholding"],
            "step5_ekstraksi":   pipeline_res["step5_ekstraksi"],
            "step6_klasifikasi": pipeline_res["step6_klasifikasi"],
            "output_10_citra":   output_10_citra,
            "dataset_10": {
                "feature_names": lc.FEATURE_NAMES,
                "records": dataset_10
            }
        }

        # Simpan ke riwayat in-memory
        try:
            riwayat_id = _simpan_riwayat_memory(
                pipeline_res=pipeline_res,
                source_name=source_name,
                leaf_name=leaf_name,
                metode_threshold=metode_threshold,
                dataset_10=dataset_10
            )
            response_data["riwayat_id"] = riwayat_id
        except Exception as mem_err:
            print(f"[RIWAYAT WARNING] {mem_err}")
            response_data["riwayat_id"] = None

        return jsonify(make_json_safe(response_data))

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# ROUTE API: RIWAYAT KLASIFIKASI (In-Memory)
# ---------------------------------------------------------------------------
@app.route("/api/riwayat", methods=["GET"])
def get_riwayat():
    """Mengambil daftar riwayat klasifikasi dari memori (50 terbaru)."""
    try:
        limit = int(request.args.get("limit", 50))
        data = []
        for r in _riwayat_store[:limit]:
            data.append({
                "id":                   r["id"],
                "waktu":                r["waktu"],
                "nama_file":            r["nama_file"],
                "nama_daun_input":      r["nama_daun_input"],
                "metode_threshold":     r["metode_threshold"],
                "daun_teridentifikasi": r["daun_teridentifikasi"],
                "nama_ilmiah":          r["nama_ilmiah"],
                "famili":               r["famili"],
                "persentase_total":     r["persentase_total"],
                "persentase_warna":     r["persentase_warna"],
                "persentase_bentuk":    r["persentase_bentuk"],
                "persentase_tekstur":   r["persentase_tekstur"],
                "status_kemiripan":     r["status_kemiripan"],
                "hex_warna":            r["hex_warna"],
                "kategori_warna":       r["kategori_warna"],
                "area_px":              r["area_px"],
                "metric":               r["metric"],
                "eccentricity":         r["eccentricity"],
                "venation_density":     r["venation_density"],
                "dimensi_asli":         r["dimensi_asli"],
            })
        return jsonify({"success": True, "data": data, "total": len(data)})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/riwayat/<int:riwayat_id>", methods=["GET"])
def get_detail_riwayat(riwayat_id):
    """Mengambil detail satu riwayat beserta dataset fiturnya."""
    try:
        record = next((r for r in _riwayat_store if r["id"] == riwayat_id), None)
        if record is None:
            return jsonify({"success": False, "error": "Riwayat tidak ditemukan"}), 404
        detail = {k: v for k, v in record.items() if k != "dataset_fitur"}
        return jsonify({
            "success": True,
            "data": {
                "riwayat": detail,
                "dataset_fitur": record.get("dataset_fitur", [])
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


@app.route("/api/riwayat/<int:riwayat_id>", methods=["DELETE"])
def delete_riwayat(riwayat_id):
    """Menghapus satu riwayat dari memori."""
    global _riwayat_store
    before = len(_riwayat_store)
    _riwayat_store = [r for r in _riwayat_store if r["id"] != riwayat_id]
    if len(_riwayat_store) == before:
        return jsonify({"success": False, "error": "Riwayat tidak ditemukan"}), 404
    return jsonify({"success": True, "message": f"Riwayat #{riwayat_id} berhasil dihapus"})


@app.route("/api/statistik", methods=["GET"])
def get_statistik():
    """Mengambil statistik ringkasan dari seluruh riwayat in-memory."""
    try:
        total = len(_riwayat_store)
        if total > 0:
            vals = [r["persentase_total"] for r in _riwayat_store if r.get("persentase_total") is not None]
            avg_sim = round(sum(vals) / len(vals), 1) if vals else 0.0
        else:
            avg_sim = 0.0

        from collections import Counter
        count = Counter(r["daun_teridentifikasi"] for r in _riwayat_store if r.get("daun_teridentifikasi"))
        distribusi = [{"daun_teridentifikasi": k, "jumlah": v} for k, v in count.most_common()]

        return jsonify({
            "success": True,
            "data": {
                "total_analisis": total,
                "rata_rata_kemiripan": avg_sim,
                "distribusi_daun": distribusi
            }
        })
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500


# ---------------------------------------------------------------------------
# Entry point untuk pengembangan lokal (bukan untuk Vercel)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
