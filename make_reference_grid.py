#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
Referans Veri Hazırlama Yardımcısı (Örneklem Kareleri)

Bu script, ortomozaik sınırları içinden RASTGELE 3 adet 30x30 m örneklem
karesi seçer ve grid_kareleri.shp olarak kaydeder. Rastgelelik sabit
seed (42) ile üretildiği için sonuçlar TEKRARLANABİLİRDİR (tezde belirtin).

Kullanıcı bu karelerin içindeki TÜM çam ağaçlarını QGIS'te elle çizerek
referans veri setini (referans_agaclar.shp) oluşturur. Bu referans,
accuracy_assessment.py ile modelin doğruluğunu ölçmek için kullanılır.

Kullanım:
    python make_reference_grid.py

Gereksinimler:
    - rasterio, geopandas, shapely, numpy
    - Ortomozaik GeoTIFF dosyası

Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
Tarih: 2024
"""

import os
import sys
import warnings

import numpy as np

# Uyarıları sustur
warnings.filterwarnings('ignore')

#==============================================================================
# KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
#==============================================================================

CONFIG = {
    # Ortomozaik GeoTIFF dosyası (Pix4D çıktısı)
    "orthomosaic_path": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_transparent_mosaic_group1.tif",

    # Çıktı klasörü (grid_kareleri.shp buraya kaydedilecek)
    "output_dir": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference",

    # Örneklem karesi kenar uzunluğu (metre)
    "kare_boyutu_m": 30.0,

    # Seçilecek kare sayısı
    "kare_sayisi": 3,

    # Rastgelelik tohumu (SABİT: tekrarlanabilirlik için değiştirmeyin)
    "seed": 42,

    # Bir karenin kabul edilmesi için izin verilen en fazla boş (NoData) oranı
    # (şeffaf mozaiklerin kenarlarındaki veri içermeyen kareleri elemek için)
    "max_bos_oran": 0.5,

    # En fazla deneme sayısı (uygun kare bulunamazsa güvenlik sınırı)
    "max_deneme": 2000,
}

#==============================================================================


def dosya_kontrol():
    """Gerekli dosyaların varlığını kontrol eder."""
    print("\n📁 Dosya kontrolü yapılıyor...")

    if not os.path.exists(CONFIG["orthomosaic_path"]):
        print(f"❌ Ortomozaik dosyası bulunamadı: {CONFIG['orthomosaic_path']}")
        print("💡 Lütfen CONFIG bölümündeki dosya yolunu kontrol edin.")
        sys.exit(1)

    print(f"   ✅ Ortomozaik: {CONFIG['orthomosaic_path']}")

    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"   ✅ Çıktı klasörü: {CONFIG['output_dir']}")


def kare_bos_orani(src, kare_geom) -> float:
    """
    Bir örneklem karesi içindeki boş (NoData/siyah) piksel oranını hesaplar.
    Hız için kare, düşük çözünürlükte (64x64) okunur.

    Args:
        src: Açık rasterio dataset (ortomozaik)
        kare_geom: Shapely box (UTM koordinatlı kare)

    Returns:
        0.0 - 1.0 arasında boş piksel oranı
    """
    from rasterio.windows import from_bounds

    minx, miny, maxx, maxy = kare_geom.bounds

    try:
        window = from_bounds(minx, miny, maxx, maxy, transform=src.transform)

        # Düşük çözünürlükte oku (64x64 yeterli, çok hızlı)
        veri = src.read(1, window=window, out_shape=(64, 64), boundless=True, fill_value=0)

        nodata = src.nodata
        if nodata is not None:
            bos = (veri == nodata) | (veri == 0)
        else:
            bos = (veri == 0)

        return float(np.mean(bos))

    except Exception:
        # Okunamayan kare tamamen boş kabul edilir
        return 1.0


def kareleri_sec(src) -> list:
    """
    Ortomozaik sınırları içinden rastgele, birbiriyle örtüşmeyen ve
    yeterli veri içeren örneklem kareleri seçer (seed=42, tekrarlanabilir).

    Returns:
        List of Shapely box geometrileri
    """
    from shapely.geometry import box

    bounds = src.bounds
    kare = CONFIG["kare_boyutu_m"]

    genislik = bounds.right - bounds.left
    yukseklik = bounds.top - bounds.bottom

    print(f"\n🗺️ Ortomozaik kapsamı: {genislik:.1f} m x {yukseklik:.1f} m")

    if genislik < kare or yukseklik < kare:
        print(f"❌ HATA: Ortomozaik {kare:.0f}x{kare:.0f} m kare için çok küçük!")
        sys.exit(1)

    # Sabit seed ile rastgele sayı üreteci (tekrarlanabilirlik)
    rng = np.random.default_rng(CONFIG["seed"])

    secilen = []
    deneme = 0

    print(f"\n🎲 Rastgele {CONFIG['kare_sayisi']} adet {kare:.0f}x{kare:.0f} m kare seçiliyor (seed={CONFIG['seed']})...")

    while len(secilen) < CONFIG["kare_sayisi"] and deneme < CONFIG["max_deneme"]:
        deneme += 1

        # Kare tamamen sınırlar içinde kalacak şekilde sol-alt köşe seç
        x = bounds.left + rng.uniform(0, genislik - kare)
        y = bounds.bottom + rng.uniform(0, yukseklik - kare)

        aday = box(x, y, x + kare, y + kare)

        # Daha önce seçilen karelerle örtüşmesin
        if any(aday.intersects(k) for k in secilen):
            continue

        # Yeterli veri içersin (kenar boşluklarını ele)
        bos_oran = kare_bos_orani(src, aday)
        if bos_oran > CONFIG["max_bos_oran"]:
            continue

        secilen.append(aday)
        print(f"   ✅ Kare {len(secilen)}: sol-alt köşe ({x:.1f}, {y:.1f}), dolu oran: %{(1-bos_oran)*100:.0f}")

    if len(secilen) < CONFIG["kare_sayisi"]:
        print(f"\n⚠️ UYARI: {CONFIG['max_deneme']} denemede sadece {len(secilen)} uygun kare bulundu.")
        print("   'max_bos_oran' değerini artırmayı deneyin (örn. 0.7).")
        if not secilen:
            sys.exit(1)

    return secilen


def grid_olustur():
    """Ana fonksiyon: kareleri seçer ve shapefile olarak kaydeder."""
    import rasterio
    import geopandas as gpd

    print("\n" + "="*60)
    print("🌲 REFERANS VERİ HAZIRLAMA - ÖRNEKLEM KARELERİ")
    print("="*60)

    dosya_kontrol()

    with rasterio.open(CONFIG["orthomosaic_path"]) as src:
        if src.crs is None:
            print("❌ HATA: Ortomozaiğin koordinat sistemi (CRS) tanımlı değil!")
            sys.exit(1)

        print(f"   🌍 CRS: {src.crs}")

        kareler = kareleri_sec(src)
        crs = src.crs

    # GeoDataFrame oluştur ve kaydet
    gdf = gpd.GeoDataFrame(
        {
            "kare_id": range(1, len(kareler) + 1),
            "alan_m2": [k.area for k in kareler],
        },
        geometry=kareler,
        crs=crs,
    )

    shp_path = os.path.join(CONFIG["output_dir"], "grid_kareleri.shp")
    gdf.to_file(shp_path, driver="ESRI Shapefile", encoding="utf-8")

    print(f"\n💾 Kaydedildi: {shp_path}")

    # Kullanıcıya QGIS talimatları
    print("\n" + "="*60)
    print("📋 ŞİMDİ NE YAPMALISINIZ? (QGIS'TE REFERANS ÇİZİMİ)")
    print("="*60)
    print("""
1. QGIS'i açın ve şu katmanları ekleyin:
   - Ortomozaik GeoTIFF  (Layer → Add Layer → Add Raster Layer)
   - grid_kareleri.shp   (Layer → Add Layer → Add Vector Layer)

2. Yeni bir poligon shapefile oluşturun:
   - Layer → Create Layer → New Shapefile Layer
   - Dosya adı : referans_agaclar.shp
   - Konum     : {output_dir}
   - Geometri  : Polygon
   - CRS       : Ortomozaikle AYNI olmalı ({crs})

3. Her örneklem karesinin İÇİNDEKİ TÜM çam ağaçlarının taçlarını
   tek tek poligon olarak çizin (Toggle Editing → Add Polygon Feature).
   - Taç sınırını olabildiğince gerçeğe yakın çizin.
   - Kare sınırında kalan ağaçlarda, taç merkezi kare İÇİNDEYSE çizin.
   - Hiçbir çamı atlamayın; referans TAM olmalı.

4. Düzenlemeyi kaydedin (Toggle Editing → Save).

5. Ardından doğruluk değerlendirmesini çalıştırın:
   python accuracy_assessment.py
""".format(output_dir=CONFIG["output_dir"], crs=crs))
    print("="*60)
    print(f"\n✅ Tamamlandı! {len(kareler)} örneklem karesi oluşturuldu.")
    print(f"   (seed={CONFIG['seed']} sabit olduğu için script yeniden çalıştırılırsa aynı kareler seçilir)")


if __name__ == "__main__":
    try:
        grid_olustur()
    except KeyboardInterrupt:
        print("\n\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
