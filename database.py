"""
=============================================================================
DATABASE.PY - MANAJEMEN DATABASE SQLite
=============================================================================
Modul ini mengelola koneksi dan operasi database SQLite untuk menyimpan:
  1. Riwayat Klasifikasi Daun (tabel: riwayat_klasifikasi)
  2. Dataset Fitur 10 Baris per Gambar (tabel: dataset_fitur)
=============================================================================
"""

import sqlite3
import os
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "daun_klasifikasi.db")


def get_connection():
    """Membuka koneksi ke database SQLite."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row  # Agar hasil query bisa diakses seperti dict
    return conn


def init_db():
    """
    Inisialisasi database: membuat tabel jika belum ada.
    Dipanggil sekali saat server Flask pertama kali berjalan.
    """
    conn = get_connection()
    cursor = conn.cursor()

    # -------------------------------------------------------------------------
    # TABEL 1: RIWAYAT KLASIFIKASI
    # Menyimpan setiap hasil analisis citra daun
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS riwayat_klasifikasi (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            waktu               TEXT NOT NULL,
            nama_file           TEXT NOT NULL,
            nama_daun_input     TEXT,
            metode_threshold    TEXT,

            -- Hasil Klasifikasi (Langkah 6)
            daun_teridentifikasi    TEXT,
            nama_ilmiah             TEXT,
            famili                  TEXT,
            persentase_total        REAL,
            persentase_warna        REAL,
            persentase_bentuk       REAL,
            persentase_tekstur      REAL,
            status_kemiripan        TEXT,

            -- Fitur Warna (Langkah 5 - Warna)
            R_mean      REAL,
            G_mean      REAL,
            B_mean      REAL,
            R_std       REAL,
            G_std       REAL,
            B_std       REAL,
            H_mean      REAL,
            S_mean      REAL,
            V_mean      REAL,
            hex_warna   TEXT,
            kategori_warna TEXT,

            -- Fitur Bentuk (Langkah 5 - Bentuk)
            area_px         REAL,
            perimeter_px    REAL,
            metric          REAL,
            eccentricity    REAL,
            aspect_ratio    REAL,
            solidity        REAL,
            extent          REAL,
            jumlah_objek    INTEGER,

            -- Fitur Tekstur (Langkah 5 - Tekstur)
            kontras         REAL,
            dissimilarity   REAL,
            homogenitas     REAL,
            energi          REAL,
            korelasi        REAL,
            venation_density REAL,

            -- Dimensi Citra
            dimensi_asli    TEXT,
            dimensi_standar TEXT
        )
    """)

    # -------------------------------------------------------------------------
    # TABEL 2: DATASET FITUR 10 BARIS
    # Menyimpan 10 variasi fitur per sesi analisis
    # -------------------------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS dataset_fitur (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            riwayat_id          INTEGER NOT NULL,
            no_variasi          INTEGER,
            nama_variasi        TEXT,
            nama_daun           TEXT,

            -- Fitur Warna
            R_mean      REAL,
            G_mean      REAL,
            B_mean      REAL,
            R_std       REAL,
            G_std       REAL,
            B_std       REAL,
            H_mean      REAL,
            S_mean      REAL,
            V_mean      REAL,

            -- Fitur Bentuk
            area_px         REAL,
            perimeter_px    REAL,
            metric          REAL,
            eccentricity    REAL,
            aspect_ratio    REAL,
            solidity        REAL,
            extent          REAL,

            -- Fitur Tekstur
            kontras             REAL,
            dissimilarity       REAL,
            homogenitas         REAL,
            energi              REAL,
            korelasi            REAL,
            venation_density    REAL,
            jumlah_objek        INTEGER,

            FOREIGN KEY (riwayat_id) REFERENCES riwayat_klasifikasi(id) ON DELETE CASCADE
        )
    """)

    conn.commit()
    conn.close()
    print(f"[DB] Database SQLite siap: {DB_PATH}")


def simpan_hasil_klasifikasi(pipeline_res, source_name, leaf_name, metode_threshold, dataset_10):
    """
    Menyimpan hasil analisis pipeline lengkap ke database.

    Parameters:
        pipeline_res     : dict hasil dari lc.proses_citra_lengkap()
        source_name      : nama file asli yang diupload
        leaf_name        : nama daun yang diekstrak
        metode_threshold : metode thresholding yang dipilih
        dataset_10       : list 10 baris fitur dari lc.buat_dataset_10_fitur()

    Returns:
        int: ID baris yang baru disimpan (riwayat_id)
    """
    waktu_sekarang = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    step1  = pipeline_res.get("step1_input", {})
    step5  = pipeline_res.get("step5_ekstraksi", {})
    step6  = pipeline_res.get("step6_klasifikasi", {})
    warna  = step5.get("warna", {})
    bentuk = step5.get("bentuk", {})
    tekstur = step5.get("tekstur", {})

    # Ambil top match (rank 1)
    daftar = step6.get("daftar_perbandingan", [])
    top = daftar[0] if daftar else {}

    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO riwayat_klasifikasi (
            waktu, nama_file, nama_daun_input, metode_threshold,
            daun_teridentifikasi, nama_ilmiah, famili,
            persentase_total, persentase_warna, persentase_bentuk, persentase_tekstur,
            status_kemiripan,
            R_mean, G_mean, B_mean, R_std, G_std, B_std,
            H_mean, S_mean, V_mean, hex_warna, kategori_warna,
            area_px, perimeter_px, metric, eccentricity, aspect_ratio, solidity, extent, jumlah_objek,
            kontras, dissimilarity, homogenitas, energi, korelasi, venation_density,
            dimensi_asli, dimensi_standar
        ) VALUES (
            ?, ?, ?, ?,
            ?, ?, ?,
            ?, ?, ?, ?,
            ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?, ?, ?,
            ?, ?, ?, ?, ?, ?,
            ?, ?
        )
    """, (
        waktu_sekarang, source_name, leaf_name, metode_threshold,

        step6.get("predicted_class"), step6.get("scientific_name"), step6.get("family"),
        top.get("persentase_total"), top.get("persentase_warna"),
        top.get("persentase_bentuk"), top.get("persentase_tekstur"),
        top.get("status_label"),

        warna.get("R_mean"), warna.get("G_mean"), warna.get("B_mean"),
        warna.get("R_std"),  warna.get("G_std"),  warna.get("B_std"),
        warna.get("H_mean"), warna.get("S_mean"), warna.get("V_mean"),
        warna.get("hex"), warna.get("kategori_warna"),

        bentuk.get("main_area"), bentuk.get("main_perimeter"), bentuk.get("metric"),
        bentuk.get("eccentricity"), bentuk.get("aspect_ratio"), bentuk.get("solidity"),
        bentuk.get("extent"), bentuk.get("jumlah_objek"),

        tekstur.get("kontras"), tekstur.get("dissimilarity"), tekstur.get("homogenitas"),
        tekstur.get("energi"),  tekstur.get("korelasi"),       tekstur.get("venation_density"),

        step1.get("orig_dim"), step1.get("dim_standard")
    ))

    riwayat_id = cursor.lastrowid

    # Simpan 10 baris dataset fitur
    for row in dataset_10:
        cursor.execute("""
            INSERT INTO dataset_fitur (
                riwayat_id, no_variasi, nama_variasi, nama_daun,
                R_mean, G_mean, B_mean, R_std, G_std, B_std,
                H_mean, S_mean, V_mean,
                area_px, perimeter_px, metric, eccentricity, aspect_ratio, solidity, extent,
                kontras, dissimilarity, homogenitas, energi, korelasi, venation_density, jumlah_objek
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?,
                ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?,
                ?, ?, ?, ?, ?, ?, ?
            )
        """, (
            riwayat_id, row.get("no"), row.get("variasi"), row.get("citra"),
            row.get("R_mean"), row.get("G_mean"), row.get("B_mean"),
            row.get("R_std"),  row.get("G_std"),  row.get("B_std"),
            row.get("H_mean"), row.get("S_mean"), row.get("V_mean"),
            row.get("area_px"), row.get("perimeter_px"), row.get("metric"),
            row.get("eccentricity"), row.get("aspect_ratio"), row.get("solidity"), row.get("extent"),
            row.get("kontras"), row.get("dissimilarity"), row.get("homogenitas"),
            row.get("energi"),  row.get("korelasi"), row.get("venation_density"),
            row.get("jumlah_objek")
        ))

    conn.commit()
    conn.close()
    print(f"[DB] Tersimpan → riwayat_id={riwayat_id} | {leaf_name} | {waktu_sekarang}")
    return riwayat_id


def ambil_semua_riwayat(limit=50):
    """
    Mengambil semua riwayat klasifikasi, diurutkan dari yang terbaru.

    Parameters:
        limit : jumlah maksimal baris yang diambil (default 50)

    Returns:
        list of dict
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, waktu, nama_file, nama_daun_input, metode_threshold,
               daun_teridentifikasi, nama_ilmiah, famili,
               persentase_total, persentase_warna, persentase_bentuk, persentase_tekstur,
               status_kemiripan, hex_warna, kategori_warna,
               area_px, metric, eccentricity, venation_density,
               dimensi_asli
        FROM riwayat_klasifikasi
        ORDER BY id DESC
        LIMIT ?
    """, (limit,))
    rows = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return rows


def ambil_detail_riwayat(riwayat_id):
    """
    Mengambil detail lengkap satu riwayat beserta 10 baris dataset fiturnya.

    Parameters:
        riwayat_id : ID dari tabel riwayat_klasifikasi

    Returns:
        dict berisi 'riwayat' dan 'dataset_fitur'
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT * FROM riwayat_klasifikasi WHERE id = ?", (riwayat_id,))
    riwayat = cursor.fetchone()
    if not riwayat:
        conn.close()
        return None

    cursor.execute("""
        SELECT * FROM dataset_fitur WHERE riwayat_id = ? ORDER BY no_variasi ASC
    """, (riwayat_id,))
    fitur_rows = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "riwayat": dict(riwayat),
        "dataset_fitur": fitur_rows
    }


def hapus_riwayat(riwayat_id):
    """
    Menghapus satu riwayat beserta seluruh dataset fitur terkait.

    Parameters:
        riwayat_id : ID yang akan dihapus

    Returns:
        bool: True jika berhasil dihapus
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM riwayat_klasifikasi WHERE id = ?", (riwayat_id,))
    affected = cursor.rowcount
    conn.commit()
    conn.close()
    return affected > 0


def hitung_statistik():
    """
    Menghitung statistik ringkasan dari seluruh riwayat.

    Returns:
        dict berisi total analisis, rata-rata kemiripan, dan distribusi per jenis daun
    """
    conn = get_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) as total FROM riwayat_klasifikasi")
    total = cursor.fetchone()["total"]

    cursor.execute("SELECT AVG(persentase_total) as avg_sim FROM riwayat_klasifikasi")
    avg_sim = cursor.fetchone()["avg_sim"] or 0.0

    cursor.execute("""
        SELECT daun_teridentifikasi, COUNT(*) as jumlah
        FROM riwayat_klasifikasi
        GROUP BY daun_teridentifikasi
        ORDER BY jumlah DESC
    """)
    distribusi = [dict(row) for row in cursor.fetchall()]

    conn.close()
    return {
        "total_analisis": total,
        "rata_rata_kemiripan": round(avg_sim, 1),
        "distribusi_daun": distribusi
    }
