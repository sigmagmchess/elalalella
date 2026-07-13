#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
CBS Çıktıları ve İstatistik Raporu Oluşturma Scripti

Bu script inference sonuçlarını CBS formatlarına (Shapefile, GeoJSON) dönüştürür,
DSM-DTM farkından ağaç boylarını hesaplar ve özet rapor oluşturur.

ÖNEMLİ: DSM (0.615 cm/px) ve DTM (3.08 cm/px) çözünürlükleri FARKLIDIR.
CHM hesaplanmadan önce DTM, rasterio.warp.reproject ile DSM grid'ine
bilinear yöntemle yeniden örneklenir; doğrudan array çıkarma yapılmaz.

Kullanım:
    python export_results.py

Gereksinimler:
    - inference_orthomosaic.py çıktıları (tespitler.pkl)
    - DSM ve DTM GeoTIFF dosyaları (ağaç boyu hesaplama için)

Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
Tarih: 2024
"""

import os
import sys
import pickle
import warnings
from typing import Dict, Any, Optional

import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib import rcParams
from tqdm import tqdm

# Türkçe karakter desteği
rcParams['font.family'] = 'DejaVu Sans'

# Uyarıları sustur
warnings.filterwarnings('ignore')

#==============================================================================
# KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
#==============================================================================

CONFIG = {
    # inference_orthomosaic.py çıktı klasörü (tespitler.pkl burada)
    "input_dir": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference",

    # DSM dosyası (Pix4D çıktısı, 0.615 cm/px) - Ağaç boyu hesaplama için
    "dsm_path": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_dsm.tif",

    # DTM dosyası (Pix4D çıktısı, 3.08 cm/px) - Ağaç boyu hesaplama için
    "dtm_path": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_dtm.tif",

    # Çıktı klasörü (Shapefile ve raporlar için)
    "output_dir": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference",

    # Koordinat sistemi (WGS84 / UTM Zone 35N)
    "target_crs": "EPSG:32635",

    # Boy hesaplama yöntemi: "max" (maksimum), "mean" (ortalama), "percentile_95"
    "boy_hesaplama": "max",
}

#==============================================================================


def dosya_kontrol() -> Dict[str, Any]:
    """Gerekli dosyaların varlığını kontrol eder."""
    print("\n📁 Dosya kontrolü yapılıyor...")

    hatalar = []
    bulunanlar = {}

    # Pickle dosyası (zorunlu)
    pickle_path = os.path.join(CONFIG["input_dir"], "tespitler.pkl")
    if not os.path.exists(pickle_path):
        hatalar.append(f"❌ Tespit dosyası bulunamadı: {pickle_path}")
        hatalar.append("   Önce inference_orthomosaic.py çalıştırın!")
    else:
        print(f"   ✅ Tespitler: {pickle_path}")
        bulunanlar["pickle"] = pickle_path

    # DSM dosyası (opsiyonel ama önerilen)
    if os.path.exists(CONFIG["dsm_path"]):
        print(f"   ✅ DSM: {CONFIG['dsm_path']}")
        bulunanlar["dsm"] = CONFIG["dsm_path"]
    else:
        print(f"   ⚠️ DSM bulunamadı (ağaç boyu hesaplanmayacak): {CONFIG['dsm_path']}")

    # DTM dosyası (opsiyonel ama önerilen)
    if os.path.exists(CONFIG["dtm_path"]):
        print(f"   ✅ DTM: {CONFIG['dtm_path']}")
        bulunanlar["dtm"] = CONFIG["dtm_path"]
    else:
        print(f"   ⚠️ DTM bulunamadı (ağaç boyu hesaplanmayacak): {CONFIG['dtm_path']}")

    if hatalar:
        print("\n" + "\n".join(hatalar))
        sys.exit(1)

    # Çıktı klasörünü oluştur
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"   ✅ Çıktı klasörü: {CONFIG['output_dir']}")

    return bulunanlar


def chm_hesapla(dsm_path: str, dtm_path: str) -> Optional[Any]:
    """
    Canopy Height Model (CHM) = DSM - DTM hesaplar.
    Ağaç boylarını belirlemek için kullanılır.

    KRİTİK: DSM ve DTM çözünürlükleri farklıdır (DSM ≈ 0.615 cm/px,
    DTM ≈ 3.08 cm/px). Bu yüzden DTM, rasterio.warp.reproject ile DSM'in
    grid'ine (aynı transform, aynı boyut) bilinear olarak yeniden örneklenir.
    Doğrudan array çıkarma yapılırsa şekil uyuşmazlığı hatası oluşur.
    """
    import rasterio
    from rasterio.warp import reproject, Resampling

    print("\n🌳 CHM (Canopy Height Model) hesaplanıyor...")
    print("   CHM = DSM - DTM (ağaç tepesi - zemin = ağaç boyu)")

    try:
        with rasterio.open(dsm_path) as dsm_src:
            dsm_data = dsm_src.read(1).astype(np.float32)
            dsm_transform = dsm_src.transform
            dsm_crs = dsm_src.crs
            dsm_nodata = dsm_src.nodata

            print(f"   DSM boyut: {dsm_src.width} x {dsm_src.height}")
            print(f"   DSM çözünürlük: {abs(dsm_transform.a)*100:.3f} cm/px")

            with rasterio.open(dtm_path) as dtm_src:
                print(f"   DTM orijinal boyut: {dtm_src.width} x {dtm_src.height}")
                print(f"   DTM çözünürlük: {abs(dtm_src.transform.a)*100:.3f} cm/px")

                # DTM'yi DSM grid'ine yeniden örnekle (bilinear resampling).
                # reproject, iki raster'ın transform'larını dikkate aldığı için
                # hem çözünürlük hem de kapsama (extent) farkları doğru işlenir.
                print("   🔄 DTM, DSM grid'ine yeniden örnekleniyor (bilinear)...")

                dtm_resampled = np.full(dsm_data.shape, np.nan, dtype=np.float32)

                reproject(
                    source=rasterio.band(dtm_src, 1),
                    destination=dtm_resampled,
                    src_transform=dtm_src.transform,
                    src_crs=dtm_src.crs,
                    src_nodata=dtm_src.nodata,
                    dst_transform=dsm_transform,
                    dst_crs=dsm_crs,
                    dst_nodata=np.nan,
                    resampling=Resampling.bilinear,
                )

        # DSM NoData değerlerini NaN yap (DTM NoData'sı reproject'te NaN oldu)
        if dsm_nodata is not None:
            dsm_data[dsm_data == dsm_nodata] = np.nan

        # CHM hesapla — artık iki array de aynı grid'de, güvenle çıkarılabilir
        chm_data = dsm_data - dtm_resampled

        # Negatif değerleri sıfırla (hatalı veriler)
        chm_data[chm_data < 0] = 0

        # Çok yüksek değerleri filtrele (hatalı veriler, >50m mantıksız)
        chm_data[chm_data > 50] = np.nan

        print(f"   ✅ CHM hesaplandı")
        print(f"   Min boy: {np.nanmin(chm_data):.2f} m")
        print(f"   Max boy: {np.nanmax(chm_data):.2f} m")
        print(f"   Ortalama boy: {np.nanmean(chm_data):.2f} m")

        return {
            "data": chm_data,
            "transform": dsm_transform,
            "crs": dsm_crs,
        }

    except Exception as e:
        print(f"   ❌ CHM hesaplama hatası: {str(e)}")
        return None


def poligon_icin_boy_hesapla(
    polygon,
    chm_data: np.ndarray,
    transform,
    method: str = "max"
) -> Optional[float]:
    """
    Bir poligon içindeki piksellerden ağaç boyunu hesaplar.

    Args:
        polygon: Shapely polygon
        chm_data: CHM numpy array
        transform: rasterio transform
        method: "max", "mean", veya "percentile_95"

    Returns:
        Ağaç boyu (metre) veya None
    """
    from rasterio.features import geometry_mask

    try:
        # Poligon için mask oluştur
        mask = geometry_mask(
            [polygon],
            out_shape=chm_data.shape,
            transform=transform,
            invert=True  # Poligon içi True
        )

        # Poligon içindeki değerleri al
        values = chm_data[mask]
        values = values[~np.isnan(values)]

        if len(values) == 0:
            return None

        if method == "max":
            return float(np.max(values))
        elif method == "mean":
            return float(np.mean(values))
        elif method == "percentile_95":
            return float(np.percentile(values, 95))
        else:
            return float(np.max(values))

    except Exception:
        return None


def istatistik_raporu_olustur(gdf: gpd.GeoDataFrame, output_dir: str):
    """Özet istatistik raporu ve grafikler oluşturur."""

    print("\n📊 İstatistik raporu oluşturuluyor...")

    # Rapor dosyası
    rapor_path = os.path.join(output_dir, "ozet_rapor.txt")

    with open(rapor_path, "w", encoding="utf-8") as f:
        f.write("="*60 + "\n")
        f.write("YTÜ HARİTA MÜHENDİSLİĞİ - ÇAM AĞACI TESPİT RAPORU\n")
        f.write("="*60 + "\n\n")

        f.write("GENEL İSTATİSTİKLER\n")
        f.write("-"*40 + "\n")
        f.write(f"Toplam tespit edilen ağaç sayısı: {len(gdf)}\n")
        f.write(f"Koordinat sistemi: {gdf.crs}\n\n")

        f.write("TAÇ ALANI İSTATİSTİKLERİ\n")
        f.write("-"*40 + "\n")
        f.write(f"Toplam taç alanı: {gdf['alan_m2'].sum():.2f} m²\n")
        f.write(f"Ortalama taç alanı: {gdf['alan_m2'].mean():.2f} m²\n")
        f.write(f"Minimum taç alanı: {gdf['alan_m2'].min():.2f} m²\n")
        f.write(f"Maksimum taç alanı: {gdf['alan_m2'].max():.2f} m²\n")
        f.write(f"Std sapma: {gdf['alan_m2'].std():.2f} m²\n\n")

        f.write("TAÇ ÇAPI İSTATİSTİKLERİ\n")
        f.write("-"*40 + "\n")
        f.write(f"Ortalama taç çapı: {gdf['cap_m'].mean():.2f} m\n")
        f.write(f"Minimum taç çapı: {gdf['cap_m'].min():.2f} m\n")
        f.write(f"Maksimum taç çapı: {gdf['cap_m'].max():.2f} m\n")
        f.write(f"Std sapma: {gdf['cap_m'].std():.2f} m\n\n")

        if "boy_m" in gdf.columns and gdf["boy_m"].notna().any():
            f.write("AĞAÇ BOYU İSTATİSTİKLERİ\n")
            f.write("-"*40 + "\n")
            boy_gecerli = gdf[gdf["boy_m"].notna()]["boy_m"]
            f.write(f"Boy hesaplanan ağaç sayısı: {len(boy_gecerli)}\n")
            f.write(f"Ortalama ağaç boyu: {boy_gecerli.mean():.2f} m\n")
            f.write(f"Minimum ağaç boyu: {boy_gecerli.min():.2f} m\n")
            f.write(f"Maksimum ağaç boyu: {boy_gecerli.max():.2f} m\n")
            f.write(f"Std sapma: {boy_gecerli.std():.2f} m\n\n")

        f.write("GÜVEN SKORU İSTATİSTİKLERİ\n")
        f.write("-"*40 + "\n")
        f.write(f"Ortalama güven: {gdf['conf'].mean():.3f}\n")
        f.write(f"Minimum güven: {gdf['conf'].min():.3f}\n")
        f.write(f"Maksimum güven: {gdf['conf'].max():.3f}\n\n")

        f.write("="*60 + "\n")
        f.write("Rapor otomatik olarak oluşturulmuştur.\n")
        f.write("YTÜ Harita Mühendisliği Yüksek Lisans Tezi\n")
        f.write("="*60 + "\n")

    print(f"   ✅ Rapor: {rapor_path}")

    # Grafikler
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("Çam Ağacı Tespit Analizi - YTÜ Harita Mühendisliği",
                 fontsize=14, fontweight='bold')

    # 1. Taç alanı histogramı
    ax1 = axes[0, 0]
    ax1.hist(gdf["alan_m2"], bins=30, color='forestgreen', edgecolor='darkgreen', alpha=0.7)
    ax1.set_xlabel("Taç Alanı (m²)")
    ax1.set_ylabel("Ağaç Sayısı")
    ax1.set_title("Taç Alanı Dağılımı")
    ax1.axvline(gdf["alan_m2"].mean(), color='red', linestyle='--',
                label=f'Ortalama: {gdf["alan_m2"].mean():.1f} m²')
    ax1.legend()

    # 2. Taç çapı histogramı
    ax2 = axes[0, 1]
    ax2.hist(gdf["cap_m"], bins=30, color='darkgreen', edgecolor='black', alpha=0.7)
    ax2.set_xlabel("Taç Çapı (m)")
    ax2.set_ylabel("Ağaç Sayısı")
    ax2.set_title("Taç Çapı Dağılımı")
    ax2.axvline(gdf["cap_m"].mean(), color='red', linestyle='--',
                label=f'Ortalama: {gdf["cap_m"].mean():.1f} m')
    ax2.legend()

    # 3. Ağaç boyu histogramı (varsa)
    ax3 = axes[1, 0]
    if "boy_m" in gdf.columns and gdf["boy_m"].notna().any():
        boy_data = gdf[gdf["boy_m"].notna()]["boy_m"]
        ax3.hist(boy_data, bins=30, color='saddlebrown', edgecolor='black', alpha=0.7)
        ax3.set_xlabel("Ağaç Boyu (m)")
        ax3.set_ylabel("Ağaç Sayısı")
        ax3.set_title("Ağaç Boyu Dağılımı")
        ax3.axvline(boy_data.mean(), color='red', linestyle='--',
                    label=f'Ortalama: {boy_data.mean():.1f} m')
        ax3.legend()
    else:
        ax3.text(0.5, 0.5, "Ağaç boyu verisi\nmevcut değil\n(DSM/DTM gerekli)",
                 ha='center', va='center', fontsize=12, color='gray')
        ax3.set_title("Ağaç Boyu Dağılımı")
        ax3.axis('off')

    # 4. Güven skoru dağılımı
    ax4 = axes[1, 1]
    ax4.hist(gdf["conf"], bins=20, color='steelblue', edgecolor='navy', alpha=0.7)
    ax4.set_xlabel("Güven Skoru")
    ax4.set_ylabel("Ağaç Sayısı")
    ax4.set_title("Model Güven Skoru Dağılımı")
    ax4.axvline(gdf["conf"].mean(), color='red', linestyle='--',
                label=f'Ortalama: {gdf["conf"].mean():.2f}')
    ax4.legend()

    plt.tight_layout()

    histogram_path = os.path.join(output_dir, "istatistik_grafikleri.png")
    plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"   ✅ Grafikler: {histogram_path}")


def export_calistir():
    """Ana export fonksiyonu."""

    print("\n" + "="*60)
    print("🌲 ÇAM AĞACI CBS ÇIKTILARI VE RAPOR OLUŞTURMA")
    print("="*60)

    # Dosya kontrolü
    bulunanlar = dosya_kontrol()

    # Tespitleri yükle
    print("\n📂 Tespit verileri yükleniyor...")

    with open(bulunanlar["pickle"], "rb") as f:
        data = pickle.load(f)

    gdf = data["gdf"]
    print(f"   ✅ {len(gdf)} adet tespit yüklendi")

    # ESRI Shapefile alan adları en fazla 10 karakter olabilir.
    # "confidence" (10+ karakter riski ve tutarlılık için) → "conf" olarak
    # kısaltılır; tüm alan adları 10 karakter sınırına uygundur:
    # id, conf, alan_m2, cap_m, boy_m, merkez_x, merkez_y
    if "confidence" in gdf.columns:
        gdf = gdf.rename(columns={"confidence": "conf"})

    # CRS kontrolü ve dönüşümü
    if gdf.crs is None:
        gdf = gdf.set_crs(CONFIG["target_crs"])
        print(f"   CRS atandı: {CONFIG['target_crs']}")
    elif str(gdf.crs) != CONFIG["target_crs"]:
        gdf = gdf.to_crs(CONFIG["target_crs"])
        print(f"   CRS dönüştürüldü: {CONFIG['target_crs']}")

    # CHM'den ağaç boyu hesapla
    if "dsm" in bulunanlar and "dtm" in bulunanlar:
        chm = chm_hesapla(bulunanlar["dsm"], bulunanlar["dtm"])

        if chm is not None:
            print("\n🌳 Her ağaç için boy hesaplanıyor...")

            boylar = []
            for idx, row in tqdm(gdf.iterrows(), total=len(gdf), desc="   Boy hesaplama"):
                boy = poligon_icin_boy_hesapla(
                    row.geometry,
                    chm["data"],
                    chm["transform"],
                    method=CONFIG["boy_hesaplama"]
                )
                boylar.append(boy)

            gdf["boy_m"] = boylar

            gecerli_boy = gdf["boy_m"].notna().sum()
            print(f"   ✅ {gecerli_boy}/{len(gdf)} ağaç için boy hesaplandı")
    else:
        print("\n⚠️ DSM/DTM bulunamadı, ağaç boyu hesaplanamıyor")
        gdf["boy_m"] = None

    # Öznitelikleri düzenle (Shapefile 10 karakter sınırına uygun adlar)
    if "boy_m" in gdf.columns:
        gdf = gdf[["id", "conf", "alan_m2", "cap_m", "boy_m", "merkez_x", "merkez_y", "geometry"]]
    else:
        gdf = gdf[["id", "conf", "alan_m2", "cap_m", "merkez_x", "merkez_y", "geometry"]]

    # Sayısal değerleri yuvarla
    gdf["conf"] = gdf["conf"].round(4)
    gdf["alan_m2"] = gdf["alan_m2"].round(3)
    gdf["cap_m"] = gdf["cap_m"].round(3)
    gdf["merkez_x"] = gdf["merkez_x"].round(3)
    gdf["merkez_y"] = gdf["merkez_y"].round(3)
    if "boy_m" in gdf.columns:
        gdf["boy_m"] = gdf["boy_m"].round(2)

    # Shapefile olarak kaydet
    print("\n💾 CBS dosyaları kaydediliyor...")

    shapefile_path = os.path.join(CONFIG["output_dir"], "cam_agaclari.shp")
    gdf.to_file(shapefile_path, driver="ESRI Shapefile", encoding="utf-8")
    print(f"   ✅ Shapefile: {shapefile_path}")

    # GeoJSON olarak kaydet
    geojson_path = os.path.join(CONFIG["output_dir"], "cam_agaclari.geojson")
    gdf.to_file(geojson_path, driver="GeoJSON")
    print(f"   ✅ GeoJSON: {geojson_path}")

    # CSV olarak da kaydet (öznitelik tablosu)
    csv_path = os.path.join(CONFIG["output_dir"], "cam_agaclari_ozellikler.csv")
    gdf_csv = gdf.drop(columns=["geometry"])
    gdf_csv.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"   ✅ CSV: {csv_path}")

    # İstatistik raporu oluştur
    istatistik_raporu_olustur(gdf, CONFIG["output_dir"])

    # Final özet
    print("\n" + "="*60)
    print("📊 SONUÇ ÖZETİ")
    print("="*60)
    print(f"   Toplam ağaç sayısı: {len(gdf)}")
    print(f"   Toplam taç alanı: {gdf['alan_m2'].sum():.2f} m²")
    print(f"   Ortalama taç çapı: {gdf['cap_m'].mean():.2f} m")
    if "boy_m" in gdf.columns and gdf["boy_m"].notna().any():
        print(f"   Ortalama ağaç boyu: {gdf['boy_m'].mean():.2f} m")
    print("="*60)

    print("\n✅ Tüm çıktılar başarıyla oluşturuldu!")
    print(f"📂 Çıktı klasörü: {CONFIG['output_dir']}")

    print("\n📋 Oluşturulan dosyalar:")
    print(f"   1. cam_agaclari.shp (+ .dbf, .shx, .prj) - QGIS'te açın")
    print(f"   2. cam_agaclari.geojson - Web haritalar için")
    print(f"   3. cam_agaclari_ozellikler.csv - Excel'de analiz için")
    print(f"   4. ozet_rapor.txt - Metin rapor")
    print(f"   5. istatistik_grafikleri.png - Görsel rapor")

    print("\n💡 QGIS'te görselleştirme:")
    print("   1. QGIS'i açın")
    print("   2. Ortomozaik GeoTIFF'i ekleyin (Layer → Add Raster Layer)")
    print("   3. cam_agaclari.shp'yi ekleyin (Layer → Add Vector Layer)")
    print("   4. Shapefile'ı ortomozaiğin üstüne sürükleyin")
    print("   5. Poligonları yeşil renk + şeffaf dolgu yapın")

    print("\n💡 Sonraki adım: make_reference_grid.py ile referans kareleri oluşturun")


if __name__ == "__main__":
    try:
        export_calistir()
    except KeyboardInterrupt:
        print("\n\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
