#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
Güven Eşiği Duyarlılık Analizi (Threshold Analysis)

Bu script farklı güven eşiği (confidence threshold) değerleri için tespit
sayısını ve toplam taç alanını hesaplar. Sonuçlar tezin "parametre seçimi"
bölümünde kullanılır.

Verimlilik notu: Ortomozaik üzerinde inference SADECE BİR KEZ, en düşük
eşikle (örn. 0.25) çalıştırılır. YOLO conf=0.25 ile çalıştırıldığında
0.25 ve üzeri güvenli TÜM tespitleri döndürdüğü için, daha yüksek eşikler
bu tespitlerin güven skoruna göre filtrelenmesiyle elde edilir ve her eşik
için Global NMS + alan filtresi yeniden uygulanır. Böylece 5 ayrı inference
yerine 1 inference yeterli olur; kod tekrarını önlemek için tüm fonksiyonlar
inference_orthomosaic.py'den import edilir.

Kullanım:
    python threshold_analysis.py

Gereksinimler:
    - inference_orthomosaic.py (aynı klasörde; fonksiyonlar oradan import edilir)
    - ultralytics, rasterio, shapely, pandas, matplotlib, tqdm

Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
Tarih: 2024
"""

import os
import sys
import warnings

import numpy as np

# Uyarıları sustur
warnings.filterwarnings('ignore')

# inference_orthomosaic.py'deki fonksiyonları import et (kod tekrarı YOK)
try:
    from inference_orthomosaic import (
        CONFIG as INFERENCE_CONFIG,
        cihaz_sec,
        dosya_kontrol,
        model_yukle,
        ortomozaik_bilgisi,
        ham_tespitleri_topla,
        global_nms_poligonlar,
        utm_poligonlara_donustur,
    )
except ImportError as e:
    print("❌ HATA: inference_orthomosaic.py import edilemedi!")
    print("   Bu script, inference_orthomosaic.py ile AYNI klasörde çalıştırılmalıdır.")
    print(f"   Detay: {e}")
    sys.exit(1)

#==============================================================================
# KONFİGÜRASYON - Bu değerleri kendi projenize göre düzenleyin
# (Model ve ortomozaik yolları inference_orthomosaic.py CONFIG'inden alınır)
#==============================================================================

CONFIG = {
    # Analiz edilecek güven eşiği değerleri
    "esikler": [0.25, 0.30, 0.40, 0.50, 0.60],

    # Çıktı klasörü (CSV ve grafik buraya kaydedilir)
    "output_dir": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference",
}

#==============================================================================


def grafik_ciz(df, output_dir: str):
    """
    Eşik-duyarlılık grafiğini çizer: ortak x ekseni (güven eşiği) üzerinde
    iki ayrı panel (tespit sayısı ve toplam taç alanı). İki büyüklüğün
    ölçeği farklı olduğu için tek grafikte çift y ekseni KULLANILMAZ.
    """
    import matplotlib.pyplot as plt
    from matplotlib import rcParams

    rcParams['font.family'] = 'DejaVu Sans'

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8), sharex=True)
    fig.suptitle("Güven Eşiği Duyarlılık Analizi", fontsize=13, fontweight="bold")

    for ax in (ax1, ax2):
        ax.set_axisbelow(True)
        ax.grid(True, color="#d9d9d9", linewidth=0.6)
        for kenar in ["top", "right"]:
            ax.spines[kenar].set_visible(False)

    # Panel 1: Tespit sayısı
    ax1.plot(df["esik"], df["tespit_sayisi"], color="forestgreen",
             linewidth=2, marker="o", markersize=8,
             markeredgecolor="white", markeredgewidth=1)
    ax1.set_ylabel("Tespit sayısı (adet)")
    ax1.set_title("Tespit Edilen Ağaç Sayısı", fontsize=11)

    # Her noktaya değer etiketi
    for _, satir in df.iterrows():
        ax1.annotate(f"{int(satir['tespit_sayisi'])}",
                     xy=(satir["esik"], satir["tespit_sayisi"]),
                     xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=9, color="#333333")

    # Panel 2: Toplam taç alanı
    ax2.plot(df["esik"], df["toplam_alan_m2"], color="#3b6ea5",
             linewidth=2, marker="o", markersize=8,
             markeredgecolor="white", markeredgewidth=1)
    ax2.set_ylabel("Toplam taç alanı (m²)")
    ax2.set_xlabel("Güven eşiği (confidence threshold)")
    ax2.set_title("Toplam Taç Alanı", fontsize=11)

    for _, satir in df.iterrows():
        ax2.annotate(f"{satir['toplam_alan_m2']:.0f}",
                     xy=(satir["esik"], satir["toplam_alan_m2"]),
                     xytext=(0, 9), textcoords="offset points",
                     ha="center", fontsize=9, color="#333333")

    ax2.set_xticks(df["esik"])

    plt.tight_layout()

    grafik_path = os.path.join(output_dir, "esik_analizi.png")
    plt.savefig(grafik_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"   ✅ Grafik: {grafik_path}")


def analiz_calistir():
    """Ana eşik analizi fonksiyonu."""
    import rasterio
    import pandas as pd

    print("\n" + "=" * 60)
    print("🌲 GÜVEN EŞİĞİ DUYARLILIK ANALİZİ")
    print("=" * 60)

    esikler = sorted(CONFIG["esikler"])
    min_esik = esikler[0]

    print(f"\n📋 Analiz edilecek eşikler: {esikler}")
    print(f"   Inference SADECE 1 kez, en düşük eşikle ({min_esik}) çalıştırılacak;")
    print(f"   diğer eşikler güven skoruna göre filtrelenerek hesaplanacak.")

    # Cihaz seçimi (CPU varsayılan, CUDA varsa otomatik GPU)
    device = cihaz_sec()

    # Dosya kontrolü (model + ortomozaik, inference CONFIG'inden)
    dosya_kontrol()

    os.makedirs(CONFIG["output_dir"], exist_ok=True)

    # Model yükle
    model = model_yukle(INFERENCE_CONFIG["model_path"])

    # Ortomozaiği aç ve ham tespitleri EN DÜŞÜK eşikle topla
    print(f"\n🗺️ Ortomozaik açılıyor: {INFERENCE_CONFIG['orthomosaic_path']}")

    with rasterio.open(INFERENCE_CONFIG["orthomosaic_path"]) as src:
        info = ortomozaik_bilgisi(src)

        print(f"   📐 Boyut: {info['width']} x {info['height']} piksel")
        print(f"   🌍 CRS: {info['crs']}")

        ham_tespitler = ham_tespitleri_topla(
            model=model,
            src=src,
            info=info,
            conf_esigi=min_esik,
            device=device,
        )

    if not ham_tespitler:
        print("\n⚠️ UYARI: Hiç tespit yapılamadı, analiz iptal edildi.")
        print("   Model veya ortomozaik yollarını ve eşik değerlerini kontrol edin.")
        return

    # Her eşik için: filtrele → Global NMS → UTM dönüşümü → say/topla
    sonuclar = []

    for esik in esikler:
        print("\n" + "-" * 60)
        print(f"📊 Eşik analiz ediliyor: conf = {esik}")
        print("-" * 60)

        # Güven skoruna göre filtrele
        filtreli = [t for t in ham_tespitler if t["confidence"] >= esik]
        print(f"   Eşiği geçen ham tespit: {len(filtreli)}")

        if not filtreli:
            sonuclar.append({
                "esik": esik,
                "tespit_sayisi": 0,
                "toplam_alan_m2": 0.0,
                "ortalama_alan_m2": 0.0,
            })
            print("   ⚠️ Bu eşikte hiç tespit kalmadı.")
            continue

        # Global NMS (her eşik için yeniden; kalan poligon kümesi değişir)
        nms_sonrasi = global_nms_poligonlar(filtreli, INFERENCE_CONFIG["global_nms_iou"])

        # UTM'e dönüştür + minimum alan filtresi
        utm_poligonlar = utm_poligonlara_donustur(nms_sonrasi, info)

        alanlar = np.array([p["alan_m2"] for p in utm_poligonlar])

        sonuclar.append({
            "esik": esik,
            "tespit_sayisi": len(utm_poligonlar),
            "toplam_alan_m2": round(float(alanlar.sum()) if len(alanlar) else 0.0, 2),
            "ortalama_alan_m2": round(float(alanlar.mean()) if len(alanlar) else 0.0, 2),
        })

    # Sonuç tablosu
    df = pd.DataFrame(sonuclar)

    print("\n" + "=" * 60)
    print("📊 EŞİK ANALİZİ SONUÇ TABLOSU")
    print("=" * 60)
    print(df.to_string(index=False))
    print("=" * 60)

    # CSV kaydet
    csv_path = os.path.join(CONFIG["output_dir"], "esik_analizi.csv")
    df.to_csv(csv_path, index=False, encoding="utf-8-sig")
    print(f"\n💾 Sonuçlar kaydediliyor...")
    print(f"   ✅ CSV: {csv_path}")

    # Çizgi grafik
    grafik_ciz(df, CONFIG["output_dir"])

    print("\n✅ Eşik analizi tamamlandı!")
    print(f"📂 Çıktı klasörü: {CONFIG['output_dir']}")
    print("\n💡 Tez için yorum: Eşik yükseldikçe tespit sayısı azalır (yanlış")
    print("   alarmlar elenir ama bazı gerçek ağaçlar da kaçırılır). Doğruluk")
    print("   değerlendirmesindeki F1 skorunu maksimize eden eşiği tercih edin.")


if __name__ == "__main__":
    try:
        analiz_calistir()
    except KeyboardInterrupt:
        print("\n\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
