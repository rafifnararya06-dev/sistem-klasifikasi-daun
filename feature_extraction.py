"""
=============================================================================
FEATURE_EXTRACTION.PY - MODUL PREPROCESSING, SEGMENTASI & EKSTRAKSI FITUR
=============================================================================
Mematuhi secara ketat alur pengolahan citra digital (Computer Vision):

1. Input Citra:
   - Menerima citra daun berwarna (BGR) dengan resolusi dan ukuran bervariasi.

2. Pre-processing:
   - Resize citra menjadi ukuran seragam (256 x 256 piksel) agar representasi fitur konsisten.
   - Filter Gaussian Blur (kernel 5x5) untuk menghaluskan permukaan gambar,
     menghilangkan noise minor, watermark halus, dan menyamarkan pola latar belakang.

3. Segmentasi Grayscale & Otsu Thresholding (Adaptif 3 Kondisi Background):
   - Konversi citra ke Grayscale.
   - Deteksi otomatis kondisi latar belakang pada tepian citra (Putih, Kotak-kotak, atau Hitam).
   - Penerapan Thresholding Otsu yang disesuaikan:
     * Background Putih: Inverted Otsu (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
     * Background Kotak-kotak (Checkerboard): Inverted Otsu dikombinasikan filter saturasi HSV
     * Background Hitam (Tanpa background): Standar Otsu (cv2.THRESH_BINARY + cv2.THRESH_OTSU)
   - Operasi Morfologi (Opening untuk menghapus bintik derau latar dan Closing untuk menambal rongga)
     sehingga menghasilkan Citra Biner solid yang mempertahankan bentuk daun aslinya tanpa cacat.

4. Ekstraksi Fitur (10 Dimensi):
   - Masking bitwise_and: Mengalikan citra biner ke citra warna asli agar sistem hanya membaca objek daun.
   - Ciri Bentuk: Area, Perimeter, dan Circularity (kebundaran bentuk) dari kontur biner.
   - Ciri Warna: Nilai rata-rata ruang warna HSV (H_mean, S_mean, V_mean) pada piksel daun ber-masking.
   - Ciri Tekstur: GLCM (Gray-Level Co-occurrence Matrix) menghitung Contrast, Energy, Homogeneity, dan Correlation.
=============================================================================
"""

import cv2
import numpy as np
import base64
from skimage.feature import graycomatrix, graycoprops

# 10 Nama Fitur Terstandar
FEATURE_NAMES = [
    "area",
    "perimeter",
    "circularity",
    "h_mean",
    "s_mean",
    "v_mean",
    "contrast",
    "energy",
    "homogeneity",
    "correlation"
]


def image_to_base64(img, is_bgr=True):
    """
    Mengubah numpy array gambar ke string Base64 Data URL (data:image/jpeg;base64,...)
    agar dapat ditampilkan langsung di antarmuka web (Frontend).
    """
    if img is None:
        return ""
    if is_bgr and len(img.shape) == 3:
        img_out = img
    elif len(img.shape) == 3:
        img_out = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    else:
        img_out = img  # Grayscale / Binary Mask
    _, buffer = cv2.imencode(".jpg", img_out, [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    encoded = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/jpeg;base64,{encoded}"


# =============================================================================
# TAHAP 1: INPUT & PRE-PROCESSING
# =============================================================================
def preprocess_image(img_bgr, target_size=(256, 256), blur_kernel=(5, 5)):
    """
    Tahap 2: Pre-processing:
    - Resize gambar agar ukuran seragam (256x256 piksel) ketika input citra berbeda ukuran.
    - Penghilangan noise / watermark dengan Gaussian Blur (kernel 5x5, sigma otomatis)
      untuk menghaluskan artefak tepi dan menyamarkan derau latar belakang.
    """
    # 1. Resize gambar seragam ke 256x256 menggunakan interpolasi area
    img_resized = cv2.resize(img_bgr, target_size, interpolation=cv2.INTER_AREA)

    # 2. Filter Gaussian Blur untuk peredaman noise / watermark halus
    img_blurred = cv2.GaussianBlur(img_resized, blur_kernel, 0)

    return img_resized, img_blurred


# =============================================================================
# TAHAP 2: SEGMENTASI GRAYSCALE & OTSU THRESHOLDING (3 KONDISI BACKGROUND)
# =============================================================================
def segment_background_adaptive(img_blurred):
    """
    Tahap 3: Segmentasi Grayscale dan Otsu Thresholding:
    Menyesuaikan pemisahan objek daun pada 3 kondisi background di dataset:
      1. Background Putih (Latar terang, daun lebih gelap)
      2. Background Kotak-kotak / Checkerboard transparan (Kotak putih-abu)
      3. Background Hitam / Tanpa Background (Latar gelap, daun lebih terang)
    
    Dilanjutkan operasi morfologi (Opening & Closing) untuk menghasilkan citra biner
    yang bersih, solid, dan mempertahankan bentuk daun aslinya tanpa cacat sekecil apa pun.
    """
    # Konversi citra hasil preprocessing ke Grayscale
    gray = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2GRAY)
    hsv = cv2.cvtColor(img_blurred, cv2.COLOR_BGR2HSV)

    # -------------------------------------------------------------------------
    # DETEKSI KARAKTERISTIK BACKGROUND PADA PERIMETER / BORDER KANVAS (12 PIKSEL)
    # -------------------------------------------------------------------------
    border_mask = np.zeros(gray.shape, dtype=bool)
    border_mask[:12, :] = True
    border_mask[-12:, :] = True
    border_mask[:, :12] = True
    border_mask[:, -12:] = True

    border_gray = gray[border_mask]
    mean_border_val = float(np.mean(border_gray))
    std_border_val = float(np.std(border_gray))

    # Logika Penyesuaian 3 Kondisi Background:
    if mean_border_val < 50:
        # ---------------------------------------------------------------------
        # KONDISI 1: BACKGROUND HITAM / TANPA LATAR BELAKANG
        # Karakteristik: Tepi kanvas gelap (intensitas keabuan < 50).
        # Logika Otsu: Objek daun adalah daerah terang (foreground),
        # sehingga menggunakan cv2.THRESH_BINARY + cv2.THRESH_OTSU.
        # Background hitam bernilai 0, objek daun bernilai 255.
        # ---------------------------------------------------------------------
        bg_condition = "Hitam (Tanpa Background)"
        otsu_val, raw_binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )

    elif std_border_val > 10 and mean_border_val > 130:
        # ---------------------------------------------------------------------
        # KONDISI 2: BACKGROUND KOTAK-KOTAK (CHECKERBOARD / POLA TRANSPARAN)
        # Karakteristik: Standar deviasi tepi tinggi (> 10) akibat selang-seling kotak.
        # Kotak-kotak memiliki saturasi warna netral/sangat rendah.
        # Daun memiliki saturasi klorofil atau kontras terhadap kotak.
        # Diterapkan Otsu Inverted dikombinasikan dengan deteksi saturasi HSV.
        # ---------------------------------------------------------------------
        bg_condition = "Kotak-kotak (Checkerboard)"
        otsu_val, otsu_bin = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )
        sat_channel = hsv[:, :, 1]
        sat_mask = (sat_channel > 28).astype(np.uint8) * 255
        raw_binary = cv2.bitwise_or(otsu_bin, sat_mask)

    else:
        # ---------------------------------------------------------------------
        # KONDISI 3: BACKGROUND PUTIH / TERANG
        # Karakteristik: Tepi kanvas terang dan seragam (mean > 200, std rendah).
        # Logika Otsu: Menggunakan Inverted Thresholding (cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        # Background putih terang (> T) menjadi hitam pekat (0),
        # dan objek daun (<= T) menjadi putih murni (255).
        # ---------------------------------------------------------------------
        bg_condition = "Putih / Terang"
        otsu_val, raw_binary = cv2.threshold(
            gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )

    # -------------------------------------------------------------------------
    # OPERASI MORFOLOGI (OPENING + CLOSING)
    # 1. Opening (cv2.MORPH_OPEN): Erosi lalu Dilasi dengan kernel kecil (3x3 atau 5x5)
    #    untuk menghilangkan bintik-bintik noise terisolasi pada background.
    # 2. Closing (cv2.MORPH_CLOSE): Dilasi lalu Erosi untuk menutup pori-pori/celah
    #    di dalam helai daun tanpa merusak tepi daun asli.
    # -------------------------------------------------------------------------
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    mask_open = cv2.morphologyEx(raw_binary, cv2.MORPH_OPEN, kernel_small, iterations=1)
    mask_closed = cv2.morphologyEx(mask_open, cv2.MORPH_CLOSE, kernel_medium, iterations=1)

    # -------------------------------------------------------------------------
    # PEMILIHAN KONTUR DAUN & HOLE FILLING (MEMPERTAHANKAN BENTUK ASLI DAUN)
    # Mengisi rongga interior daun agar citra biner solid dan utuh.
    # -------------------------------------------------------------------------
    contours, _ = cv2.findContours(mask_closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    clean_binary = np.zeros_like(mask_closed)

    if contours:
        # Urutkan kontur berdasarkan luas area
        sorted_cnts = sorted(contours, key=cv2.contourArea, reverse=True)
        max_area = cv2.contourArea(sorted_cnts[0])
        # Ambil kontur daun utama (luas signifikan)
        valid_cnts = [c for c in sorted_cnts if cv2.contourArea(c) > max(200.0, max_area * 0.05)]
        cv2.drawContours(clean_binary, valid_cnts, -1, 255, thickness=-1)
    else:
        clean_binary = mask_closed

    return gray, clean_binary, float(otsu_val), bg_condition


# =============================================================================
# TAHAP 3: EKSTRAKSI FITUR BENTUK, WARNA, DAN TEKSTUR (LENGKAP & VISUALISASI)
# =============================================================================
def extract_detailed_shape_features(img_resized, clean_binary):
    """
    Ekstraksi Domain Bentuk (Morfometri Fisik Daun):
      - Area: Luas bidang piksel daun (cv2.contourArea)
      - Perimeter: Keliling tepi kontur daun (cv2.arcLength)
      - Circularity (Kebundaran): 4 * pi * Area / Perimeter^2
      - Eccentricity (Kelonjongan): Eksentrisitas elips e = sqrt(1 - (b/a)^2)
      - Aspect Ratio: Rasio lebar terhadap panjang bounding box (min/max)
      - Solidity: Kepadatan rasio Area kontur terhadap Area Convex Hull
      - Extent: Rasio Area daun terhadap Luas Bounding Box
      - Visualisasi Morfometri: Citra daun dengan Kontur Utama (Hijau),
        Convex Hull (Oranye), dan Bounding Box (Cyan/Kuning).
    """
    contours, _ = cv2.findContours(clean_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        img_morph = img_resized.copy()
        return {
            "area": 0.0,
            "perimeter": 0.0,
            "circularity": 0.0,
            "eccentricity": 0.0,
            "aspect_ratio": 1.0,
            "solidity": 0.0,
            "extent": 0.0,
            "main_contour": None,
            "visual_morphometry": image_to_base64(img_morph, is_bgr=True)
        }

    main_contour = max(contours, key=cv2.contourArea)
    area = float(cv2.contourArea(main_contour))
    perimeter = float(cv2.arcLength(main_contour, True))
    circularity = float((4.0 * np.pi * area) / (perimeter ** 2)) if perimeter > 0 else 0.0

    bx, by, bw, bh = cv2.boundingRect(main_contour)
    aspect_ratio = float(min(bw, bh)) / float(max(bw, bh)) if max(bw, bh) > 0 else 1.0
    extent = float(area) / float(bw * bh) if (bw * bh) > 0 else 0.0

    hull = cv2.convexHull(main_contour)
    hull_area = float(cv2.contourArea(hull))
    solidity = float(area) / float(hull_area) if hull_area > 0 else 0.0

    eccentricity = 0.0
    if len(main_contour) >= 5:
        try:
            (cx, cy), (MA, ma), angle = cv2.fitEllipse(main_contour)
            major_ax = max(MA, ma) / 2.0
            minor_ax = min(MA, ma) / 2.0
            if major_ax > 0:
                eccentricity = float(np.sqrt(max(0.0, 1.0 - (minor_ax / major_ax) ** 2)))
        except Exception:
            eccentricity = 0.0

    # Visualisasi Morfometri Fisik pada Citra Asli
    img_morph = img_resized.copy()
    if hull is not None and len(hull) > 0:
        cv2.drawContours(img_morph, [hull], -1, (0, 165, 255), 2)  # Oranye: Convex Hull
    if bw > 0 and bh > 0:
        cv2.rectangle(img_morph, (bx, by), (bx + bw, by + bh), (255, 220, 0), 2)  # Cyan: Bounding Box
    cv2.drawContours(img_morph, [main_contour], -1, (0, 255, 0), 2)  # Hijau Terang: Kontur Utama

    return {
        "area": round(area, 2),
        "perimeter": round(perimeter, 2),
        "circularity": round(circularity, 4),
        "eccentricity": round(eccentricity, 4),
        "aspect_ratio": round(aspect_ratio, 4),
        "solidity": round(solidity, 4),
        "extent": round(extent, 4),
        "main_contour": main_contour,
        "visual_morphometry": image_to_base64(img_morph, is_bgr=True)
    }


def extract_detailed_color_features(img_resized, clean_binary):
    """
    Ekstraksi Domain Warna (RGB, HSV, Hex Dominan & Nama Warna Alami):
    HANYA diekstrak pada piksel helai daun ber-masking (background diabaikan).
    """
    masked_color = cv2.bitwise_and(img_resized, img_resized, mask=clean_binary)
    leaf_mask = clean_binary > 0

    if not np.any(leaf_mask):
        return {
            "masked_color": masked_color,
            "h_mean": 0.0, "s_mean": 0.0, "v_mean": 0.0,
            "h_std": 0.0, "s_std": 0.0, "v_std": 0.0,
            "r_mean": 0.0, "g_mean": 0.0, "b_mean": 0.0,
            "r_std": 0.0, "g_std": 0.0, "b_std": 0.0,
            "hex": "#2e7d32",
            "kategori_warna": "Tidak Terdeteksi"
        }

    # BGR ke RGB
    img_rgb = cv2.cvtColor(masked_color, cv2.COLOR_BGR2RGB)
    r_vals = img_rgb[:, :, 0][leaf_mask]
    g_vals = img_rgb[:, :, 1][leaf_mask]
    b_vals = img_rgb[:, :, 2][leaf_mask]

    r_mean = float(np.mean(r_vals))
    g_mean = float(np.mean(g_vals))
    b_mean = float(np.mean(b_vals))
    r_std = float(np.std(r_vals))
    g_std = float(np.std(g_vals))
    b_std = float(np.std(b_vals))

    # HSV
    hsv_img = cv2.cvtColor(masked_color, cv2.COLOR_BGR2HSV)
    h_vals = hsv_img[:, :, 0][leaf_mask]
    s_vals = hsv_img[:, :, 1][leaf_mask]
    v_vals = hsv_img[:, :, 2][leaf_mask]

    h_mean = float(np.mean(h_vals))
    s_mean = float(np.mean(s_vals))
    v_mean = float(np.mean(v_vals))
    h_std = float(np.std(h_vals))
    s_std = float(np.std(s_vals))
    v_std = float(np.std(v_vals))

    hex_color = f"#{int(np.clip(r_mean, 0, 255)):02x}{int(np.clip(g_mean, 0, 255)):02x}{int(np.clip(b_mean, 0, 255)):02x}"

    # Klasifikasi Nama Warna Alami
    if (r_mean > 160 and g_mean > 110 and b_mean < 95) or (14 <= h_mean <= 35 and s_mean > 100):
        kategori_warna = "Kuning Keemasan / Oranye Musim Gugur"
    elif (r_mean > 140 and g_mean < 100 and b_mean < 100) or ((h_mean < 12 or h_mean > 168) and s_mean > 100):
        kategori_warna = "Merah / Cokelat Kemerahan"
    elif 35 <= h_mean <= 48 and r_mean > 65 and g_mean > 115:
        kategori_warna = "Hijau Pupus / Kekuningan (Lime Green)"
    elif g_mean > r_mean and g_mean > b_mean:
        if v_mean < 95:
            kategori_warna = "Hijau Tua Beludru (Deep Green)"
        elif s_mean > 140:
            kategori_warna = "Hijau Segar Terang (Vivid Green)"
        else:
            kategori_warna = "Hijau Alami Sedang (Natural Green)"
    elif r_mean > 100 and g_mean > 80:
        kategori_warna = "Cokelat / Zaitun Alami (Olive/Brown)"
    else:
        kategori_warna = "Hijau Zaitun Alami (Olive Green)"

    return {
        "masked_color": masked_color,
        "h_mean": round(h_mean, 2), "s_mean": round(s_mean, 2), "v_mean": round(v_mean, 2),
        "h_std": round(h_std, 2), "s_std": round(s_std, 2), "v_std": round(v_std, 2),
        "r_mean": round(r_mean, 2), "g_mean": round(g_mean, 2), "b_mean": round(b_mean, 2),
        "r_std": round(r_std, 2), "g_std": round(g_std, 2), "b_std": round(b_std, 2),
        "hex": hex_color,
        "kategori_warna": kategori_warna
    }


def extract_detailed_texture_features(masked_color, clean_binary):
    """
    Ekstraksi Domain Tekstur (GLCM & X-Ray Venasi Urat Daun):
      - GLCM: Contrast, Energy, Homogeneity, Correlation, Dissimilarity.
      - Peta Urat Daun (X-Ray Map): Ekstraksi venasi pembuluh angkut dengan CLAHE + Canny Edge
        diproyeksikan pada kanvas gelap dengan pendar hijau neon.
      - Kerapatan Vena: Rasio luas urat daun terhadap total luas permukaan helai daun.
    """
    masked_gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)
    quantized_gray = (masked_gray // 16).astype(np.uint8)

    glcm = graycomatrix(
        quantized_gray,
        distances=[1],
        angles=[0],
        levels=16,
        symmetric=True,
        normed=True
    )

    contrast = float(graycoprops(glcm, "contrast")[0, 0])
    energy = float(graycoprops(glcm, "energy")[0, 0])
    homogeneity = float(graycoprops(glcm, "homogeneity")[0, 0])
    dissimilarity = float(graycoprops(glcm, "dissimilarity")[0, 0])
    try:
        correlation = float(graycoprops(glcm, "correlation")[0, 0])
        if np.isnan(correlation):
            correlation = 0.0
    except Exception:
        correlation = 0.0

    # Struktur Urat Tulang Daun (Venation Extraction menggunakan CLAHE + Canny)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    enhanced = clahe.apply(masked_gray)
    edges = cv2.Canny(enhanced, 40, 130)
    edges_masked = cv2.bitwise_and(edges, edges, mask=clean_binary)

    leaf_area_px = float(np.sum(clean_binary > 0))
    venation_density = float(np.sum(edges_masked > 0)) / float(max(1.0, leaf_area_px))

    # Peta Visual X-Ray Map Tekstur Urat Daun (Kanvas Gelap + Garis Hijau Neon Berpendar)
    h, w = masked_gray.shape[:2]
    xray_visual = np.zeros((h, w, 3), dtype=np.uint8)
    xray_visual[clean_binary > 0] = [18, 32, 22]       # Latar daun hijau sangat gelap
    xray_visual[edges_masked > 0] = [120, 245, 170]    # Garis urat neon hijau terang berpendar

    deskripsi_tekstur = "Tekstur Halus Homogen" if contrast < 0.25 else "Tekstur Berserat Kontras Tinggi"

    return {
        "contrast": round(contrast, 4),
        "energy": round(energy, 4),
        "homogeneity": round(homogeneity, 4),
        "dissimilarity": round(dissimilarity, 4),
        "correlation": round(correlation, 4),
        "venation_density": round(venation_density, 4),
        "venation_percentage": f"{round(venation_density * 100, 1)}%",
        "deskripsi_tekstur": deskripsi_tekstur,
        "visual_xray": image_to_base64(xray_visual, is_bgr=True)
    }


# =============================================================================
# PIPELINE UTAMA PENGOLAHAN CITRA
# =============================================================================
def process_leaf_pipeline(img_bgr):
    """
    Menjalankan alur lengkap pengolahan citra digital secara runtut:
      1. Pre-processing: Resize seragam (256x256) & Gaussian Blur
      2. Segmentasi: Grayscale & Otsu Thresholding adaptif (Putih, Kotak-kotak, Hitam)
      3. Ekstraksi Fitur Bentuk (Area, Perimeter, Circularity, Eccentricity, Aspect Ratio, Solidity, Morfometri Visual)
      4. Masking bitwise_and untuk isolasi daun murni
      5. Ekstraksi Fitur Warna Lengkap (RGB Mean/Std, HSV Mean/Std, Hex Dominan, Kategori Warna)
      6. Ekstraksi Fitur Tekstur GLCM & Venasi (Contrast, Energy, Homogeneity, Correlation, Dissimilarity, X-Ray Map)
    """
    # 1. Pre-processing
    img_resized, img_blurred = preprocess_image(img_bgr, target_size=(256, 256))

    # 2. Segmentasi Grayscale & Otsu Thresholding Adaptif
    gray_img, clean_binary, otsu_val, bg_condition = segment_background_adaptive(img_blurred)

    # 3. Ekstraksi Fitur Bentuk Morfometri
    shape_data = extract_detailed_shape_features(img_resized, clean_binary)
    area = shape_data["area"]
    perimeter = shape_data["perimeter"]
    circularity = shape_data["circularity"]
    main_contour = shape_data["main_contour"]

    # 4. Masking & Ekstraksi Fitur Warna Lengkap
    color_data = extract_detailed_color_features(img_resized, clean_binary)
    masked_color = color_data["masked_color"]
    h_mean = color_data["h_mean"]
    s_mean = color_data["s_mean"]
    v_mean = color_data["v_mean"]

    # 5. Ekstraksi Fitur Tekstur GLCM & Peta X-Ray Urat Daun
    texture_data = extract_detailed_texture_features(masked_color, clean_binary)
    contrast = texture_data["contrast"]
    energy = texture_data["energy"]
    homogeneity = texture_data["homogeneity"]
    correlation = texture_data["correlation"]

    # Grayscale dari citra yang sudah tersegmentasi (hanya daun)
    segmented_gray = cv2.cvtColor(masked_color, cv2.COLOR_BGR2GRAY)

    # Visualisasi kontur sederhana
    img_contour = img_resized.copy()
    if main_contour is not None:
        cv2.drawContours(img_contour, [main_contour], -1, (0, 255, 0), 2)

    # 10 Vektor Fitur Terstandar untuk Inferensi Model K-NN
    feature_vector = np.array([
        area,
        perimeter,
        circularity,
        h_mean,
        s_mean,
        v_mean,
        contrast,
        energy,
        homogeneity,
        correlation
    ], dtype=np.float32)

    return {
        "visual": {
            "citra_asli": image_to_base64(img_resized, is_bgr=True),
            "citra_preprocessing": image_to_base64(img_blurred, is_bgr=True),
            "citra_grayscale": image_to_base64(gray_img, is_bgr=False),
            "citra_biner": image_to_base64(clean_binary, is_bgr=False),
            "citra_masked": image_to_base64(masked_color, is_bgr=True),
            "citra_segmented_gray": image_to_base64(segmented_gray, is_bgr=False),
            "citra_kontur": image_to_base64(img_contour, is_bgr=True),
            "citra_xray": texture_data["visual_xray"],
            "citra_morfometri": shape_data["visual_morphometry"]
        },
        "features": {
            # Ciri Bentuk Lengkap
            "bentuk": {
                "area": area,
                "perimeter": perimeter,
                "circularity": circularity,
                "eccentricity": shape_data["eccentricity"],
                "aspect_ratio": shape_data["aspect_ratio"],
                "solidity": shape_data["solidity"],
                "extent": shape_data["extent"]
            },
            # Ciri Warna Lengkap
            "warna": {
                "h_mean": h_mean,
                "s_mean": s_mean,
                "v_mean": v_mean,
                "h_std": color_data["h_std"],
                "s_std": color_data["s_std"],
                "v_std": color_data["v_std"],
                "r_mean": color_data["r_mean"],
                "g_mean": color_data["g_mean"],
                "b_mean": color_data["b_mean"],
                "r_std": color_data["r_std"],
                "g_std": color_data["g_std"],
                "b_std": color_data["b_std"],
                "hex": color_data["hex"],
                "kategori_warna": color_data["kategori_warna"]
            },
            # Ciri Tekstur Lengkap
            "tekstur": {
                "contrast": contrast,
                "energy": energy,
                "homogeneity": homogeneity,
                "dissimilarity": texture_data["dissimilarity"],
                "correlation": correlation,
                "venation_density": texture_data["venation_density"],
                "venation_percentage": texture_data["venation_percentage"],
                "deskripsi_tekstur": texture_data["deskripsi_tekstur"]
            }
        },
        "feature_vector": feature_vector,
        "otsu_threshold": round(otsu_val, 1),
        "background_condition": bg_condition
    }
