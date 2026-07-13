# 🌲 İHA Görüntülerinden Çam Ağacı Tespiti ve Segmentasyonu

**YTÜ Harita Mühendisliği Yüksek Lisans Tezi — Python İşlem Hattı (Pipeline)**

Bu proje, İHA (drone) ile elde edilen ortomozaik üzerinde **YOLOv8-seg** derin
öğrenme modeli kullanarak çam ağaçlarını tek tek tespit eder (instance
segmentation), taç alanı/çapı ve DSM-DTM farkından ağaç boyu hesaplar,
sonuçları CBS formatlarında (Shapefile, GeoJSON) dışa aktarır ve ağaç bazlı
doğruluk değerlendirmesi yapar.

> Proje **yalnızca Python + Jupyter** içerir; web uygulaması değildir.

---

## 📦 Kurulum (Windows 11, Python 3.11)

```bash
pip install -r requirements.txt
```

GPU (CUDA 11.8) desteği için:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

> Tüm scriptler cihazı otomatik seçer: CUDA destekli GPU varsa `cuda`,
> yoksa `cpu` kullanılır ve hangi cihazın kullanıldığı konsola yazılır.

### Beklenen klasör yapısı

```
C:\Users\meryemnur.cevik\Desktop\cam_tezi\
├── 02_pix4d_outputs\
│   ├── Proses_transparent_mosaic_group1.tif   (ortomozaik)
│   ├── Proses_dsm.tif                         (DSM, 0.615 cm/px)
│   └── Proses_dtm.tif                         (DTM, 3.08 cm/px)
├── 04_training\
│   └── best.pt                                (Colab'dan indirilen model)
└── 05_inference\                              (tüm çıktılar buraya)
```

Yollar farklıysa her scriptin başındaki **CONFIG** bloğunu düzenleyin.

---

## 🗺️ İşlem Hattı — Sıralı Kullanım Kılavuzu

### Adım 1 — Roboflow'da Etiketleme

Ortomozaikten kestiğiniz karoları [Roboflow](https://roboflow.com)'a yükleyin
ve çam taçlarını **polygon** aracıyla etiketleyin (sınıf adı: `cam`).
Veri setini `YOLOv8 Segmentation` formatında versiyonlayın.

| Girdi | Çıktı |
|---|---|
| Karo görüntüleri (PNG/JPG) | Roboflow veri seti (train/valid/test, `data.yaml`) |

### Adım 2 — Colab'da Model Eğitimi

`train_yolov8seg.ipynb` dosyasını [Google Colab](https://colab.research.google.com)'da
açın (GPU çalışma zamanı seçin), Roboflow API anahtarınızı girin ve hücreleri
sırayla çalıştırın. Eğitim `seed=42` ve `deterministic=True` ile
**tekrarlanabilir** şekilde yapılır (tezde belirtin).

| Girdi | Çıktı |
|---|---|
| Roboflow veri seti | `best.pt` (Google Drive'a kaydedilir) |
| — | `egitim_metrikleri.csv` (mAP50, mAP50-95, Precision, Recall — Box/Mask) |
| — | Eğitim grafikleri (`results.png`, confusion matrix, PR eğrisi) |

### Adım 3 — best.pt Dosyasını İndirin

Google Drive'daki `YTU_Tez_Cam_Segmentation/best.pt` dosyasını bilgisayarınıza,
`04_training\best.pt` konumuna kopyalayın.

### Adım 4 — Ortomozaik Üzerinde Tahmin

```bash
python inference_orthomosaic.py
```

Ortomozaik örtüşmeli karolara bölünür, model her karoda tahmin yapar,
tespitler Global NMS ile birleştirilir ve UTM koordinatlı poligonlara dönüştürülür.

| Girdi | Çıktı |
|---|---|
| `best.pt`, ortomozaik GeoTIFF | `05_inference\cam_tespitleri.geojson` |
| — | `05_inference\tespitler.pkl` (export için ara dosya) |

### Adım 5 — CBS Çıktıları ve Ağaç Boyu

```bash
python export_results.py
```

DTM, DSM grid'ine **bilinear yöntemle yeniden örneklenir** (çözünürlükleri
farklıdır: DSM 0.615 cm/px, DTM 3.08 cm/px) ve CHM = DSM − DTM'den her ağacın
boyu hesaplanır. Shapefile öznitelik adları ESRI'nin 10 karakter sınırına
uygundur: `id, conf, alan_m2, cap_m, boy_m, merkez_x, merkez_y`.

| Girdi | Çıktı |
|---|---|
| `tespitler.pkl`, DSM, DTM | `cam_agaclari.shp` (+ .dbf/.shx/.prj) |
| — | `cam_agaclari.geojson`, `cam_agaclari_ozellikler.csv` |
| — | `ozet_rapor.txt`, `istatistik_grafikleri.png` |

### Adım 6 — Referans Örneklem Kareleri

```bash
python make_reference_grid.py
```

Ortomozaik sınırlarından rastgele **3 adet 30×30 m** örneklem karesi seçilir
(**seed=42**, tekrarlanabilir) ve `grid_kareleri.shp` olarak kaydedilir.

| Girdi | Çıktı |
|---|---|
| Ortomozaik GeoTIFF | `05_inference\grid_kareleri.shp` |

### Adım 7 — QGIS'te Referans Çizimi (Elle)

QGIS'te ortomozaik + `grid_kareleri.shp` katmanlarını açın; her karenin
**içindeki tüm çam taçlarını** poligon olarak elle çizin ve
`05_inference\referans_agaclar.shp` olarak kaydedin (CRS ortomozaikle aynı
olmalı). Ayrıntılı talimat, Adım 6'nın konsol çıktısında yazdırılır.

| Girdi | Çıktı |
|---|---|
| Ortomozaik, `grid_kareleri.shp` | `05_inference\referans_agaclar.shp` |

### Adım 8 — Ağaç Bazlı Doğruluk Değerlendirmesi

```bash
python accuracy_assessment.py
```

Tahmin ve referans poligonları **Macar algoritması** ile birebir eşleştirilir;
**IoU ≥ 0.5** eşleşmeler TP, eşleşmeyen tahminler FP, eşleşmeyen referanslar FN
sayılır. Ağaç bazında Precision / Recall / F1 ve eşleşen çiftler için taç alanı
karşılaştırması (R², RMSE) hesaplanır. **"Model kaç ağacı doğru buldu?"**
sorusunun cevabı bu rapordadır.

| Girdi | Çıktı |
|---|---|
| `cam_agaclari.shp`, `referans_agaclar.shp`, (`grid_kareleri.shp`) | `accuracy_report.txt` |
| — | `alan_karsilastirma_sacilim.png`, `eslesen_ciftler.csv` |

### Adım 9 — Güven Eşiği Duyarlılık Analizi

```bash
python threshold_analysis.py
```

`conf = 0.25, 0.30, 0.40, 0.50, 0.60` eşikleri için tespit sayısı ve toplam
taç alanı hesaplanır (inference **tek sefer** çalışır, eşikler filtrelenir;
fonksiyonlar `inference_orthomosaic.py`'den import edilir). Sonuç, tezin
**parametre seçimi** bölümüne girer.

| Girdi | Çıktı |
|---|---|
| `best.pt`, ortomozaik GeoTIFF | `esik_analizi.csv` |
| — | `esik_analizi.png` (eşik–tespit sayısı / toplam alan grafikleri) |

---

## 📄 Proje Dosyaları

| Dosya | Görev |
|---|---|
| `train_yolov8seg.ipynb` | Colab'da YOLOv8m-seg eğitimi (seed=42, deterministic) |
| `inference_orthomosaic.py` | Ortomozaik üzerinde karo bazlı tahmin |
| `export_results.py` | Shapefile/GeoJSON/CSV çıktıları + CHM'den ağaç boyu |
| `make_reference_grid.py` | 3 adet 30×30 m referans örneklem karesi (seed=42) |
| `accuracy_assessment.py` | Ağaç bazlı Precision/Recall/F1 + alan karşılaştırması |
| `threshold_analysis.py` | Güven eşiği duyarlılık analizi (CSV + grafik) |
| `requirements.txt` | Python bağımlılıkları |
| `docs/proje_dokumantasyon.html` | Tüm kodların tek sayfalık HTML dokümantasyonu |

## 💡 Notlar

- Tüm scriptlerin başında **CONFIG** bloğu vardır; yolları oradan değiştirin.
- Tüm çıktı mesajları ve raporlar **Türkçe**dir.
- Koordinat sistemi varsayılanı **EPSG:32635** (WGS84 / UTM 35N)'tir;
  farklı bölge için `export_results.py` içindeki `target_crs`'yi güncelleyin.
- Tekrarlanabilirlik: eğitimde `seed=42, deterministic=True`; örneklem
  karelerinde `seed=42` kullanılır (tezin yöntem bölümünde belirtin).
