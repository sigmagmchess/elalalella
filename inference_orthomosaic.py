#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
Ortomozaik Üzerinde Inference (Tahmin) Scripti

Bu script büyük GeoTIFF ortomozaiği karolara bölerek YOLOv8-seg modeli ile
tahmin yapar ve sonuçları georeferanslı poligon olarak çıkarır.

Buradaki fonksiyonlar threshold_analysis.py tarafından da import edilir;
bu yüzden ana akış (inference_calistir) parametreli fonksiyonlara ayrılmıştır.

Kullanım:
    python inference_orthomosaic.py

Gereksinimler:
    - Python 3.11+
    - ultralytics, rasterio, shapely, geopandas, tqdm
    - Eğitilmiş best.pt model dosyası
    - Ortomozaik GeoTIFF dosyası

Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
Tarih: 2024
"""

import os
import sys
import warnings
from typing import List, Tuple, Dict, Any

import numpy as np
from tqdm import tqdm

# Uyarıları sustur
warnings.filterwarnings('ignore')

#==============================================================================
# KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
#==============================================================================

CONFIG = {
    # Model dosyası (Colab'dan indirdiğiniz best.pt)
    "model_path": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\04_training\best.pt",

    # Ortomozaik GeoTIFF dosyası (Pix4D çıktısı)
    "orthomosaic_path": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_transparent_mosaic_group1.tif",

    # Çıktı klasörü (oluşturulacak)
    "output_dir": r"C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference",

    # Karo boyutu (piksel) - GPU belleğine göre ayarlayın
    "tile_size": 1280,

    # Örtüşme oranı (%20 = 0.2)
    "overlap_ratio": 0.2,

    # Tahmin güven eşiği (0.0-1.0)
    "confidence_threshold": 0.4,

    # NMS IoU eşiği (0.0-1.0)
    "iou_threshold": 0.5,

    # Global NMS IoU eşiği (karo birleştirme için)
    "global_nms_iou": 0.5,

    # Minimum alan filtresi (m²) - çok küçük tespitleri ele
    "min_area_m2": 1.0,

    # Cihaz: "cpu" varsayılan; script başında CUDA varsa otomatik "cuda" seçilir
    "device": "cpu",
}

#==============================================================================


def cihaz_sec() -> str:
    """
    Kullanılacak cihazı belirler: CUDA destekli GPU varsa otomatik olarak
    "cuda"ya geçer, yoksa CONFIG'deki varsayılan "cpu" kullanılır.
    Hangi cihazın kullanıldığını konsola yazar.
    """
    import torch

    if torch.cuda.is_available():
        device = "cuda"
        print(f"🖥️ Cihaz: CUDA GPU bulundu → {torch.cuda.get_device_name(0)}")
    else:
        device = "cpu"
        print("🖥️ Cihaz: GPU bulunamadı, CPU kullanılacak (işlem daha yavaş olabilir)")

    CONFIG["device"] = device
    return device


def dosya_kontrol():
    """Gerekli dosyaların varlığını kontrol eder."""
    print("\n📁 Dosya kontrolü yapılıyor...")

    hatalar = []

    if not os.path.exists(CONFIG["model_path"]):
        hatalar.append(f"❌ Model dosyası bulunamadı: {CONFIG['model_path']}")
    else:
        print(f"   ✅ Model: {CONFIG['model_path']}")

    if not os.path.exists(CONFIG["orthomosaic_path"]):
        hatalar.append(f"❌ Ortomozaik dosyası bulunamadı: {CONFIG['orthomosaic_path']}")
    else:
        size_gb = os.path.getsize(CONFIG["orthomosaic_path"]) / (1024**3)
        print(f"   ✅ Ortomozaik: {CONFIG['orthomosaic_path']} ({size_gb:.2f} GB)")

    if hatalar:
        print("\n" + "\n".join(hatalar))
        print("\n💡 Lütfen CONFIG bölümündeki dosya yollarını kontrol edin.")
        sys.exit(1)

    # Çıktı klasörünü oluştur
    os.makedirs(CONFIG["output_dir"], exist_ok=True)
    print(f"   ✅ Çıktı klasörü: {CONFIG['output_dir']}")


def model_yukle(model_path: str):
    """YOLOv8-seg modelini yükler ve döndürür."""
    from ultralytics import YOLO

    print(f"\n📦 Model yükleniyor: {model_path}")
    model = YOLO(model_path)
    print("   ✅ Model yüklendi")
    return model


def ortomozaik_bilgisi(src) -> Dict[str, Any]:
    """Ortomozaik hakkında bilgi döndürür."""
    info = {
        "width": src.width,
        "height": src.height,
        "bands": src.count,
        "crs": str(src.crs),
        "transform": src.transform,
        "bounds": src.bounds,
        "pixel_size_x": abs(src.transform.a),
        "pixel_size_y": abs(src.transform.e),
    }
    return info


def karo_pencerelerini_olustur(
    width: int,
    height: int,
    tile_size: int,
    overlap: float
) -> List[Tuple[int, int, int, int]]:
    """
    Örtüşmeli karo pencereleri oluşturur.

    Returns:
        List of (x_offset, y_offset, tile_width, tile_height)
    """
    stride = int(tile_size * (1 - overlap))
    pencereler = []

    y = 0
    while y < height:
        x = 0
        while x < width:
            # Karo boyutunu sınırla (kenarlar için)
            w = min(tile_size, width - x)
            h = min(tile_size, height - y)

            # Çok küçük karoları atla
            if w >= tile_size // 4 and h >= tile_size // 4:
                pencereler.append((x, y, w, h))

            x += stride
        y += stride

    return pencereler


def piksel_to_utm(
    piksel_coords: np.ndarray,
    transform
) -> np.ndarray:
    """
    Piksel koordinatlarını UTM koordinatlarına dönüştürür.

    Args:
        piksel_coords: (N, 2) array of (col, row) pixel coordinates
        transform: rasterio affine transform

    Returns:
        (N, 2) array of (x, y) UTM coordinates
    """
    import rasterio

    cols = piksel_coords[:, 0]
    rows = piksel_coords[:, 1]

    xs, ys = rasterio.transform.xy(transform, rows, cols)

    return np.column_stack([xs, ys])


def mask_to_polygon(mask: np.ndarray, offset_x: int, offset_y: int):
    """
    Binary mask'ı Shapely poligonuna dönüştürür.

    Args:
        mask: 2D binary numpy array
        offset_x, offset_y: Karonun piksel offsetleri

    Returns:
        Shapely Polygon veya None
    """
    from shapely.geometry import Polygon
    import cv2

    # Kontur bul
    mask_uint8 = (mask > 0.5).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return None

    # En büyük konturu al
    contour = max(contours, key=cv2.contourArea)

    if len(contour) < 3:
        return None

    # Kontur noktalarını düzleştir ve offset ekle
    points = contour.squeeze()
    if len(points.shape) == 1:
        return None

    # Offset ekle (global piksel koordinatları)
    points = points.astype(float)
    points[:, 0] += offset_x
    points[:, 1] += offset_y

    try:
        polygon = Polygon(points)
        if polygon.is_valid and polygon.area > 0:
            return polygon
        else:
            # Geçersiz poligonu düzeltmeyi dene
            polygon = polygon.buffer(0)
            if polygon.is_valid and polygon.area > 0:
                return polygon
    except:
        pass

    return None


def ham_tespitleri_topla(
    model,
    src,
    info: Dict[str, Any],
    conf_esigi: float,
    device: str,
    tile_size: int = None,
    overlap_ratio: float = None,
    iou_threshold: float = None,
) -> List[Dict]:
    """
    Ortomozaiği karolara bölerek modelle tarar ve ham tespitleri
    (piksel koordinatlı poligon + güven skoru) toplar.

    threshold_analysis.py bu fonksiyonu düşük bir eşikle (örn. 0.25) bir kez
    çağırıp, daha yüksek eşikleri güven skoruna göre filtreleyerek analiz eder.

    Args:
        model: Yüklenmiş YOLO modeli
        src: Açık rasterio dataset (ortomozaik)
        info: ortomozaik_bilgisi() çıktısı
        conf_esigi: Tahmin güven eşiği
        device: "cuda" veya "cpu"
        tile_size, overlap_ratio, iou_threshold: None ise CONFIG'den alınır

    Returns:
        List of {"polygon": Polygon (piksel), "confidence": float, ...}
    """
    from rasterio.windows import Window
    import cv2

    tile_size = tile_size if tile_size is not None else CONFIG["tile_size"]
    overlap_ratio = overlap_ratio if overlap_ratio is not None else CONFIG["overlap_ratio"]
    iou_threshold = iou_threshold if iou_threshold is not None else CONFIG["iou_threshold"]

    # Karo pencerelerini oluştur
    pencereler = karo_pencerelerini_olustur(
        info["width"],
        info["height"],
        tile_size,
        overlap_ratio
    )

    print(f"\n📦 Toplam karo sayısı: {len(pencereler)}")
    print(f"   Karo boyutu: {tile_size}px")
    print(f"   Örtüşme: %{int(overlap_ratio*100)}")
    print(f"   Güven eşiği: {conf_esigi}")

    tum_tespitler = []

    print("\n🔍 Inference başlıyor...")

    for x_off, y_off, w, h in tqdm(pencereler, desc="   Karolar işleniyor"):
        try:
            # Karoyu oku (sadece RGB bantları)
            window = Window(x_off, y_off, w, h)

            if info["bands"] >= 3:
                tile_data = src.read([1, 2, 3], window=window)
            else:
                tile_data = src.read(1, window=window)
                tile_data = np.stack([tile_data, tile_data, tile_data])

            # (C, H, W) -> (H, W, C)
            tile_rgb = np.transpose(tile_data, (1, 2, 0))

            # NoData kontrolü (tamamen boş karoları atla)
            if np.all(tile_rgb == 0) or np.all(tile_rgb == 255):
                continue

            # uint8'e dönüştür
            if tile_rgb.dtype != np.uint8:
                tile_rgb = np.clip(tile_rgb, 0, 255).astype(np.uint8)

            # YOLO tahmin
            results = model.predict(
                source=tile_rgb,
                conf=conf_esigi,
                iou=iou_threshold,
                imgsz=tile_size,
                device=device,
                verbose=False,
                retina_masks=True,  # Yüksek çözünürlüklü mask
            )

            result = results[0]

            # Tespit yoksa atla
            if result.masks is None or len(result.masks) == 0:
                continue

            # Her tespit için poligon oluştur
            masks = result.masks.data.cpu().numpy()
            boxes = result.boxes

            for idx in range(len(masks)):
                mask = masks[idx]
                conf = float(boxes.conf[idx])

                # Mask'ı orijinal karo boyutuna yeniden boyutlandır
                mask_resized = cv2.resize(
                    mask.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_LINEAR
                )

                # Poligona dönüştür
                polygon = mask_to_polygon(mask_resized, x_off, y_off)

                if polygon is not None:
                    tum_tespitler.append({
                        "polygon": polygon,
                        "confidence": conf,
                        "karo_x": x_off,
                        "karo_y": y_off,
                    })

        except Exception:
            # Hatalı karoları sessizce atla
            continue

    print(f"\n📊 Ham tespit sayısı: {len(tum_tespitler)}")

    return tum_tespitler


def global_nms_poligonlar(
    poligonlar: List[Dict],
    iou_threshold: float
) -> List[Dict]:
    """
    Tüm karolardaki poligonları birleştirir ve mükerrer tespitleri eler.
    IoU tabanlı Non-Maximum Suppression uygular.

    Args:
        poligonlar: List of {"polygon": Polygon, "confidence": float, ...}
        iou_threshold: IoU eşiği (bu değerin üstündeki örtüşmeler birleştirilir)

    Returns:
        Filtrelenmiş poligon listesi
    """
    if not poligonlar:
        return []

    print(f"\n🔄 Global NMS uygulanıyor ({len(poligonlar)} poligon)...")

    # Güven skoruna göre sırala (yüksekten düşüğe)
    poligonlar = sorted(poligonlar, key=lambda x: x["confidence"], reverse=True)

    secilen = []
    elenen_indexler = set()

    for i, p1 in enumerate(tqdm(poligonlar, desc="   NMS işlemi")):
        if i in elenen_indexler:
            continue

        secilen.append(p1)
        poly1 = p1["polygon"]

        # Bu poligonla örtüşenleri ele
        for j in range(i + 1, len(poligonlar)):
            if j in elenen_indexler:
                continue

            poly2 = poligonlar[j]["polygon"]

            try:
                # IoU hesapla
                intersection = poly1.intersection(poly2).area
                union = poly1.union(poly2).area

                if union > 0:
                    iou = intersection / union
                    if iou > iou_threshold:
                        elenen_indexler.add(j)
            except:
                continue

    print(f"   ✅ NMS sonrası: {len(secilen)} poligon (elenen: {len(elenen_indexler)})")

    return secilen


def utm_poligonlara_donustur(
    tespitler: List[Dict],
    info: Dict[str, Any],
    min_area_m2: float = None,
) -> List[Dict]:
    """
    Piksel koordinatlı tespit poligonlarını UTM koordinatlarına dönüştürür
    ve minimum alan filtresini uygular.

    Args:
        tespitler: List of {"polygon": Polygon (piksel), "confidence": float}
        info: ortomozaik_bilgisi() çıktısı
        min_area_m2: Minimum taç alanı (None ise CONFIG'den alınır)

    Returns:
        List of {"geometry": Polygon (UTM), "confidence": float, "alan_m2": float}
    """
    from shapely.geometry import Polygon

    min_area_m2 = min_area_m2 if min_area_m2 is not None else CONFIG["min_area_m2"]

    print("\n🌍 Koordinat dönüşümü yapılıyor (Piksel → UTM)...")

    utm_poligonlar = []

    for tespit in tqdm(tespitler, desc="   Dönüştürülüyor"):
        poly = tespit["polygon"]

        # Poligon köşelerini al
        coords = np.array(poly.exterior.coords)

        # UTM'e dönüştür
        utm_coords = piksel_to_utm(coords, info["transform"])

        try:
            utm_poly = Polygon(utm_coords)

            if utm_poly.is_valid and utm_poly.area > 0:
                # Alan filtresini uygula
                alan_m2 = utm_poly.area

                if alan_m2 >= min_area_m2:
                    utm_poligonlar.append({
                        "geometry": utm_poly,
                        "confidence": tespit["confidence"],
                        "alan_m2": alan_m2,
                    })
        except:
            continue

    print(f"   ✅ Geçerli UTM poligon sayısı: {len(utm_poligonlar)}")

    return utm_poligonlar


def inference_calistir():
    """Ana inference fonksiyonu."""
    import rasterio
    import geopandas as gpd

    print("\n" + "="*60)
    print("🌲 ÇAM AĞACI INSTANCE SEGMENTATION - ORTOMOZAIK INFERENCE")
    print("="*60)

    # Cihaz seçimi (CPU varsayılan, CUDA varsa otomatik GPU)
    device = cihaz_sec()

    # Dosya kontrolü
    dosya_kontrol()

    # Model yükle
    model = model_yukle(CONFIG["model_path"])

    # Ortomozaiği aç
    print(f"\n🗺️ Ortomozaik açılıyor: {CONFIG['orthomosaic_path']}")

    with rasterio.open(CONFIG["orthomosaic_path"]) as src:
        info = ortomozaik_bilgisi(src)

        print(f"   📐 Boyut: {info['width']} x {info['height']} piksel")
        print(f"   🌍 CRS: {info['crs']}")
        print(f"   📏 Piksel boyutu: {info['pixel_size_x']*100:.2f} cm/px")
        print(f"   🔲 Bant sayısı: {info['bands']}")

        # Karoları tara ve ham tespitleri topla
        tum_tespitler = ham_tespitleri_topla(
            model=model,
            src=src,
            info=info,
            conf_esigi=CONFIG["confidence_threshold"],
            device=device,
        )

        if not tum_tespitler:
            print("\n⚠️ UYARI: Hiç tespit yapılamadı!")
            print("   Olası nedenler:")
            print("   - Güven eşiği çok yüksek (confidence_threshold'u düşürün)")
            print("   - Model bu veriye uygun değil")
            print("   - Ortomozaikte çam ağacı yok")
            return

        # Global NMS uygula
        filtrelenmis = global_nms_poligonlar(
            tum_tespitler,
            CONFIG["global_nms_iou"]
        )

        # Piksel koordinatlarını UTM'e dönüştür
        utm_poligonlar = utm_poligonlara_donustur(filtrelenmis, info)

        if not utm_poligonlar:
            print("\n⚠️ UYARI: UTM dönüşümü sonrası geçerli poligon kalmadı!")
            return

        # GeoDataFrame oluştur
        print("\n💾 Sonuçlar kaydediliyor...")

        gdf = gpd.GeoDataFrame(
            utm_poligonlar,
            crs=info["crs"],
            geometry="geometry"
        )

        # Ek öznitelikler hesapla
        gdf["id"] = range(1, len(gdf) + 1)

        # Merkez koordinatları
        gdf["merkez_x"] = gdf.geometry.centroid.x
        gdf["merkez_y"] = gdf.geometry.centroid.y

        # Eşdeğer daire çapı
        gdf["cap_m"] = 2 * np.sqrt(gdf["alan_m2"] / np.pi)

        # Sütun sırasını düzenle
        gdf = gdf[["id", "confidence", "alan_m2", "cap_m", "merkez_x", "merkez_y", "geometry"]]

        # GeoJSON olarak kaydet
        geojson_path = os.path.join(CONFIG["output_dir"], "cam_tespitleri.geojson")
        gdf.to_file(geojson_path, driver="GeoJSON")
        print(f"   ✅ GeoJSON: {geojson_path}")

        # Pickle olarak da kaydet (export_results.py için)
        import pickle
        pickle_path = os.path.join(CONFIG["output_dir"], "tespitler.pkl")
        with open(pickle_path, "wb") as f:
            pickle.dump({
                "gdf": gdf,
                "crs": str(info["crs"]),
                "transform": info["transform"],
                "bounds": info["bounds"],
                "pixel_size": info["pixel_size_x"],
            }, f)
        print(f"   ✅ Pickle: {pickle_path}")

        # Özet rapor
        print("\n" + "="*60)
        print("📊 INFERENCE SONUÇ ÖZETİ")
        print("="*60)
        print(f"   Toplam tespit: {len(gdf)}")
        print(f"   Toplam taç alanı: {gdf['alan_m2'].sum():.2f} m²")
        print(f"   Ortalama taç alanı: {gdf['alan_m2'].mean():.2f} m²")
        print(f"   Ortalama taç çapı: {gdf['cap_m'].mean():.2f} m")
        print(f"   Min güven: {gdf['confidence'].min():.3f}")
        print(f"   Max güven: {gdf['confidence'].max():.3f}")
        print(f"   Ortalama güven: {gdf['confidence'].mean():.3f}")
        print("="*60)

        print("\n✅ Inference tamamlandı!")
        print(f"📂 Çıktı klasörü: {CONFIG['output_dir']}")
        print("\n💡 Sonraki adım: export_results.py çalıştırın")


if __name__ == "__main__":
    try:
        inference_calistir()
    except KeyboardInterrupt:
        print("\n\n⚠️ İşlem kullanıcı tarafından iptal edildi.")
        sys.exit(0)
    except Exception as e:
        print(f"\n❌ HATA: {str(e)}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
