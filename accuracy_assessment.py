#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
Ağaç Bazlı Doğruluk Değerlendirmesi (Accuracy Assessment)

Bu script, modelin ürettiği tahmin poligonlarını (cam_agaclari.shp)
kullanıcının QGIS'te elle çizdiği referans poligonlarla
(referans_agaclar.shp) karşılaştırır ve "model kaç ağacı doğru buldu?"
sorusuna cevap verir.

Yöntem:
    - Tahmin-referans çiftleri Macar algoritması
      (scipy.optimize.linear_sum_assignment) ile BİREBİR eşleştirilir.
    - IoU >= 0.5 olan eşleşmeler True Positive (TP) sayılır.
    - Eşleşmeyen tahminler False Positive (FP),
      eşleşmeyen referanslar False Negative (FN) olur.
    - Precision, Recall ve F1 ağaç bazında hesaplanır.
    - Eşleşen çiftler için taç alanı karşılaştırması yapılır
      (R², RMSE, saçılım grafiği).

Kullanım:
    python accuracy_assessment.py

Gereksinimler:
    - geopandas, shapely, scipy, matplotlib, tqdm
    - cam_agaclari.shp (export_results.py çıktısı)
    - referans_agaclar.shp (QGIS'te elle çizilen referans)

Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
Tarih: 2024
"""

import os
import sys
import warnings

import numpy as np
from tqdm import tqdm

# Uyarıları sustur
warnings.filterwarnings('ignore')

#==============================================================================
# KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
#==============================================================================

CONFIG = {
    # Modelin ürettiği tahmin poligonları (export_results.py çıktısı)
    "tahmin_shp": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference\cam_agaclari.shp",

    # Referans poligonlar (QGIS'te elle çizilen kontrol verisi)
    "referans_shp": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference\referans_agaclar.shp",

    # Örneklem kareleri (make_reference_grid.py çıktısı, OPSİYONEL).
    # Dosya varsa tahminler bu karelerle sınırlandırılır; çünkü referans
    # sadece kareler içinde çizildiği için tüm sahadaki tahminlerle
    # karşılaştırmak Precision'ı haksız yere düşürür.
    "grid_shp": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference\grid_kareleri.shp",

    # Eşleşme için minimum IoU eşiği (tez literatüründe standart: 0.5)
    "iou_esigi": 0.5,

    # Çıktı klasörü (rapor ve grafik buraya kaydedilir)
    "output_dir": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference",
}

#==============================================================================


def dosya_kontrol():
    """Gerekli dosyaların varlığını kontrol eder."""
    print("\n📁 Dosya kontrolü yapılıyor...")

    hatalar = []

    if not os.path.exists(CONFIG["tahmin_shp"]):
        hatalar.append(f"❌ Tahmin shapefile bulunamadı: {CONFIG['tahmin_shp']}")
        hatalar.append("   Önce inference_orthomosaic.py ve export_results.py çalıştırın!")
    else:
        print(f"   ✅ Tahminler: {CONFIG['tahmin_shp']}")

    if not os.path.exists(CONFIG["referans_shp"]):
        hatalar.append(f"❌ Referans shapefile bulunamadı: {CONFIG['referans_shp']}")
        hatalar.append("   QGIS'te referans poligonları çizip bu yola kaydedin.")
        hatalar.append("   (Yardım için: python make_reference_grid.py)")
    else:
        print(f"   ✅ Referans: {CONFIG['referans_shp']}")

    if hatalar:
        print("\n" + "\n".join(hatalar))
        sys.exit(1)

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"   ✅ Çıktı klasörü: {CONFIG['output_dir']}")


def geometri_duzelt(gdf):
    """Geçersiz geometrileri buffer(0) ile düzeltir, boşları eler."""
    gdf = gdf[gdf.geometry.notna()].copy()
    gecersiz = ~gdf.geometry.is_valid
    if gecersiz.any():
        gdf.loc[gecersiz, "geometry"] = gdf.loc[gecersiz, "geometry"].buffer(0)
    gdf = gdf[gdf.geometry.area > 0]
    return gdf


def iou_hesapla(poly1, poly2) -> float:
    """İki poligon arasındaki IoU (Intersection over Union) değerini hesaplar."""
    try:
        if not poly1.intersects(poly2):
            return 0.0
        kesisim = poly1.intersection(poly2).area
        birlesim = poly1.union(poly2).area
        if birlesim <= 0:
            return 0.0
        return kesisim / birlesim
    except Exception:
        return 0.0


def eslestir(tahminler, referanslar, iou_esigi: float):
    """
    Tahmin ve referans poligonlarını Macar algoritması ile BİREBİR eşleştirir.

    Args:
        tahminler: Tahmin geometrileri listesi
        referanslar: Referans geometrileri listesi
        iou_esigi: TP sayılmak için minimum IoU

    Returns:
        eslesmeler: List of (tahmin_index, referans_index, iou) — sadece IoU >= eşik
        iou_matrisi: (n_tahmin, n_referans) IoU matrisi
    """
    from scipy.optimize import linear_sum_assignment

    n_t, n_r = len(tahminler), len(referanslar)

    print(f"\n🔗 Eşleştirme yapılıyor ({n_t} tahmin x {n_r} referans)...")

    # IoU matrisini hesapla
    iou_matrisi = np.zeros((n_t, n_r), dtype=np.float64)

    for i in tqdm(range(n_t), desc="   IoU matrisi"):
        for j in range(n_r):
            iou_matrisi[i, j] = iou_hesapla(tahminler[i], referanslar[j])

    # Macar algoritması: toplam IoU'yu MAKSİMİZE eden birebir atama
    # (linear_sum_assignment maliyeti minimize eder, bu yüzden -IoU verilir)
    satirlar, sutunlar = linear_sum_assignment(-iou_matrisi)

    # Sadece IoU eşiğini geçen atamalar gerçek eşleşme sayılır
    eslesmeler = []
    for i, j in zip(satirlar, sutunlar):
        iou = iou_matrisi[i, j]
        if iou >= iou_esigi:
            eslesmeler.append((int(i), int(j), float(iou)))

    print(f"   ✅ IoU >= {iou_esigi} olan eşleşme sayısı: {len(eslesmeler)}")

    return eslesmeler, iou_matrisi


def alan_karsilastirma(eslesmeler, tahminler, referanslar, output_dir: str):
    """
    Eşleşen çiftler için taç alanı karşılaştırması yapar:
    R², RMSE ve saçılım grafiği (tahmin vs referans).

    Returns:
        dict: {"r2": ..., "rmse": ..., "n": ...} veya None (eşleşme yoksa)
    """
    from scipy import stats
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['font.family'] = 'DejaVu Sans'

    if len(eslesmeler) < 2:
        print("\n⚠️ Alan karşılaştırması için en az 2 eşleşme gerekli, atlanıyor.")
        return None

    print("\n📐 Taç alanı karşılaştırması yapılıyor (tahmin vs referans)...")

    ref_alan = np.array([referanslar[j].area for _, j, _ in eslesmeler])
    tah_alan = np.array([tahminler[i].area for i, _, _ in eslesmeler])

    # Doğrusal regresyon ile R² ve RMSE
    reg = stats.linregress(ref_alan, tah_alan)
    r2 = float(reg.rvalue ** 2)
    rmse = float(np.sqrt(np.mean((tah_alan - ref_alan) ** 2)))

    print(f"   R²   : {r2:.4f}")
    print(f"   RMSE : {rmse:.3f} m²")

    # Saçılım grafiği (tek seri; 1:1 doğrusu referans çizgisi olarak)
    fig, ax = plt.subplots(figsize=(7, 7))

    ax.set_axisbelow(True)
    ax.grid(True, color="#d9d9d9", linewidth=0.6)
    for kenar in ["top", "right"]:
        ax.spines[kenar].set_visible(False)

    ax.scatter(ref_alan, tah_alan, s=45, color="forestgreen",
               edgecolor="white", linewidth=0.8, zorder=3)

    # 1:1 doğrusu (mükemmel uyum çizgisi)
    maksimum = max(ref_alan.max(), tah_alan.max()) * 1.05
    ax.plot([0, maksimum], [0, maksimum], linestyle="--",
            color="#8c8c8c", linewidth=1.5, zorder=2)
    ax.annotate("1:1 doğrusu", xy=(maksimum * 0.82, maksimum * 0.86),
                color="#666666", fontsize=10, rotation=45,
                ha="center", va="center")

    ax.set_xlim(0, maksimum)
    ax.set_ylim(0, maksimum)
    ax.set_aspect("equal")

    ax.set_xlabel("Referans taç alanı (m²)")
    ax.set_ylabel("Tahmin edilen taç alanı (m²)")
    ax.set_title("Taç Alanı: Model Tahmini vs Referans", fontweight="bold")

    # İstatistikleri grafik üzerine yaz
    ax.text(0.03, 0.97,
            f"n = {len(eslesmeler)}\nR² = {r2:.3f}\nRMSE = {rmse:.2f} m²",
            transform=ax.transAxes, va="top", ha="left", fontsize=10,
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#cccccc"))

    plt.tight_layout()

    grafik_path = os.path.join(output_dir, "alan_karsilastirma_sacilim.png")
    plt.savefig(grafik_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Saçılım grafiği: {grafik_path}")

    return {"r2": r2, "rmse": rmse, "n": len(eslesmeler),
            "ref_toplam": float(ref_alan.sum()), "tah_toplam": float(tah_alan.sum())}


def rapor_yaz(sonuclar: dict, output_dir: str):
    """Sonuçları hem konsola hem accuracy_report.txt dosyasına Türkçe yazar."""

    satirlar = []
    satirlar.append("=" * 60)
    satirlar.append("YTÜ HARİTA MÜHENDİSLİĞİ - AĞAÇ BAZLI DOĞRULUK RAPORU")
    satirlar.append("=" * 60)
    satirlar.append("")
    satirlar.append("VERİ ÖZETİ")
    satirlar.append("-" * 40)
    satirlar.append(f"Tahmin poligon sayısı  : {sonuclar['n_tahmin']}")
    satirlar.append(f"Referans poligon sayısı: {sonuclar['n_referans']}")
    if sonuclar.get("grid_kullanildi"):
        satirlar.append("Not: Tahminler örneklem kareleri (grid_kareleri.shp) ile")
        satirlar.append("     sınırlandırılmıştır; referans yalnızca bu karelerde çizilidir.")
    satirlar.append("")
    satirlar.append(f"EŞLEŞTİRME SONUÇLARI (IoU eşiği = {sonuclar['iou_esigi']})")
    satirlar.append("-" * 40)
    satirlar.append(f"True Positive  (doğru tespit)        : {sonuclar['tp']}")
    satirlar.append(f"False Positive (yanlış alarm)        : {sonuclar['fp']}")
    satirlar.append(f"False Negative (kaçırılan ağaç)      : {sonuclar['fn']}")
    satirlar.append("")
    satirlar.append("DOĞRULUK METRİKLERİ (AĞAÇ BAZINDA)")
    satirlar.append("-" * 40)
    satirlar.append(f"Precision (kesinlik)  : {sonuclar['precision']:.4f}  (%{sonuclar['precision']*100:.1f})")
    satirlar.append(f"Recall    (duyarlılık): {sonuclar['recall']:.4f}  (%{sonuclar['recall']*100:.1f})")
    satirlar.append(f"F1 skoru              : {sonuclar['f1']:.4f}  (%{sonuclar['f1']*100:.1f})")
    satirlar.append("")
    if sonuclar["eslesmeler"]:
        ort_iou = np.mean([e[2] for e in sonuclar["eslesmeler"]])
        satirlar.append(f"Eşleşen çiftlerin ortalama IoU değeri : {ort_iou:.4f}")
        satirlar.append("")

    if sonuclar.get("alan") is not None:
        alan = sonuclar["alan"]
        satirlar.append("TAÇ ALANI KARŞILAŞTIRMASI (EŞLEŞEN ÇİFTLER)")
        satirlar.append("-" * 40)
        satirlar.append(f"Karşılaştırılan çift sayısı : {alan['n']}")
        satirlar.append(f"R² (belirleme katsayısı)    : {alan['r2']:.4f}")
        satirlar.append(f"RMSE                        : {alan['rmse']:.3f} m²")
        satirlar.append(f"Toplam referans alanı       : {alan['ref_toplam']:.2f} m²")
        satirlar.append(f"Toplam tahmin alanı         : {alan['tah_toplam']:.2f} m²")
        satirlar.append("Saçılım grafiği             : alan_karsilastirma_sacilim.png")
        satirlar.append("")

    satirlar.append("=" * 60)
    satirlar.append("Rapor otomatik olarak oluşturulmuştur.")
    satirlar.append("YTÜ Harita Mühendisliği Yüksek Lisans Tezi")
    satirlar.append("=" * 60)

    rapor_metni = "\n".join(satirlar)

    # Konsola yaz
    print("\n" + rapor_metni)

    # Dosyaya yaz
    rapor_path = os.path.join(output_dir, "accuracy_report.txt")
    with open(rapor_path, "w", encoding="utf-8") as f:
        f.write(rapor_metni + "\n")

    print(f"\n✅ Rapor kaydedildi: {rapor_path}")


def degerlendirme_calistir():
    """Ana doğruluk değerlendirme fonksiyonu."""
    import geopandas as gpd
    import pandas as pd

    print("\n" + "=" * 60)
    print("🌲 AĞAÇ BAZLI DOĞRULUK DEĞERLENDİRMESİ")
    print("=" * 60)

    dosya_kontrol()

    # Verileri yükle
    print("\n📂 Shapefile'lar yükleniyor...")
    tahmin_gdf = gpd.read_file(CONFIG["tahmin_shp"])
    referans_gdf = gpd.read_file(CONFIG["referans_shp"])

    print(f"   ✅ Tahmin: {len(tahmin_gdf)} poligon")
    print(f"   ✅ Referans: {len(referans_gdf)} poligon")

    if len(referans_gdf) == 0:
        print("\n❌ HATA: Referans shapefile boş! QGIS'te poligon çizdiğinizden emin olun.")
        sys.exit(1)

    # CRS uyumu: referansı tahmin CRS'ine dönüştür
    if referans_gdf.crs != tahmin_gdf.crs:
        print(f"   🔄 Referans CRS dönüştürülüyor: {referans_gdf.crs} → {tahmin_gdf.crs}")
        referans_gdf = referans_gdf.to_crs(tahmin_gdf.crs)

    # Geometrileri temizle
    tahmin_gdf = geometri_duzelt(tahmin_gdf)
    referans_gdf = geometri_duzelt(referans_gdf)

    # Örneklem kareleri varsa tahminleri karelerle sınırla
    # (referans sadece kareler içinde çizildiği için adil karşılaştırma)
    grid_kullanildi = False
    if CONFIG["grid_shp"] and os.path.exists(CONFIG["grid_shp"]):
        grid_gdf = gpd.read_file(CONFIG["grid_shp"])
        if grid_gdf.crs != tahmin_gdf.crs:
            grid_gdf = grid_gdf.to_crs(tahmin_gdf.crs)

        kareler_birlesik = grid_gdf.geometry.union_all() if hasattr(grid_gdf.geometry, "union_all") \
            else grid_gdf.geometry.unary_union

        onceki = len(tahmin_gdf)
        # Taç merkezi kare içinde olan tahminler değerlendirilir
        tahmin_gdf = tahmin_gdf[tahmin_gdf.geometry.centroid.within(kareler_birlesik)]
        grid_kullanildi = True

        print(f"\n🔲 Örneklem kareleri bulundu: {CONFIG['grid_shp']}")
        print(f"   Tahminler karelerle sınırlandırıldı: {onceki} → {len(tahmin_gdf)}")
    else:
        print("\n🔲 Örneklem karesi dosyası yok; TÜM tahminler değerlendirilecek.")
        print("   (Referans tüm sahayı kapsamıyorsa Precision olduğundan düşük çıkar!)")

    tahminler = list(tahmin_gdf.geometry)
    referanslar = list(referans_gdf.geometry)

    if len(tahminler) == 0:
        print("\n❌ HATA: Değerlendirilecek tahmin poligonu kalmadı!")
        print("   Grid kareleri ile tahminlerin aynı bölgede olduğundan emin olun.")
        sys.exit(1)

    # Macar algoritması ile birebir eşleştir
    eslesmeler, _ = eslestir(tahminler, referanslar, CONFIG["iou_esigi"])

    # TP / FP / FN
    tp = len(eslesmeler)
    fp = len(tahminler) - tp
    fn = len(referanslar) - tp

    # Metrikler (sıfıra bölme koruması)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Eşleşen çiftleri CSV olarak kaydet (tez eki için)
    if eslesmeler:
        eslesme_df = pd.DataFrame([
            {
                "tahmin_no": i + 1,
                "referans_no": j + 1,
                "iou": round(iou, 4),
                "tahmin_alan_m2": round(tahminler[i].area, 3),
                "referans_alan_m2": round(referanslar[j].area, 3),
            }
            for i, j, iou in eslesmeler
        ])
        eslesme_csv = os.path.join(CONFIG["output_dir"], "eslesen_ciftler.csv")
        eslesme_df.to_csv(eslesme_csv, index=False, encoding="utf-8-sig")
        print(f"   ✅ Eşleşen çiftler: {eslesme_csv}")

    # Taç alanı karşılaştırması (R², RMSE, saçılım grafiği)
    alan_sonuc = alan_karsilastirma(eslesmeler, tahminler, referanslar, CONFIG["output_dir"])

    # Raporu yaz (konsol + accuracy_report.txt)
    rapor_yaz(
        {
            "n_tahmin": len(tahminler),
            "n_referans": len(referanslar),
            "iou_esigi": CONFIG["iou_esigi"],
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "eslesmeler": eslesmeler,
            "alan": alan_sonuc,
            "grid_kullanildi": grid_kullanildi,
        },
        CONFIG["output_dir"],
    )

    print("\n💡 Sonraki adım: threshold_analysis.py ile güven eşiği analizini yapın")


if __name__ == "__main__":
    try:
        degerlendirme_calistir()
    except KeyboardInterrupt:
        print("\n\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
