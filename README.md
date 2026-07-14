# 🌲 İHA Görüntülerinden Çam Ağacı Tespiti ve Segmentasyonu

**YTÜ Harita Mühendisliği Yüksek Lisans Tezi — JavaScript (Node.js) İşlem Hattı**

Bu proje, İHA (drone) ile elde edilen ortomozaik üzerinde **YOLOv8-seg** derin
öğrenme modeli kullanarak çam ağaçlarını tek tek tespit eder (instance
segmentation), taç alanı/çapı ve DSM-DTM farkından ağaç boyu hesaplar,
sonuçları CBS formatlarında (Shapefile, GeoJSON) dışa aktarır ve ağaç bazlı
doğruluk değerlendirmesi yapar.

> Yerel işlem hattı **tamamen JavaScript/Node.js**'tir. Tek istisna model
> eğitimidir: YOLOv8 eğitimi yalnızca Python destekler ve Google Colab'da
> `train_yolov8seg.ipynb` ile yapılır (bilgisayarınıza Python kurmanız gerekmez).
> Tüm kodlar tek sayfalık HTML dokümantasyona bağlıdır:
> **`docs/proje_dokumantasyon.html`** (tarayıcıda açın).

---

## 🧠 Çam-AI — Tarayıcıda Çalışan Mini Öğrenen Sınıflandırıcı (`mini_ai.html`)

**Kurulum gerektirmez:** `mini_ai.html` dosyasına çift tıklayın (Chrome/Edge/Firefox/Safari).
Tamamen HTML+JavaScript'tir, hiçbir veri internete gönderilmez.

Pix4D çıktınızı **doğrudan** yükleyebilirsiniz:

| Desteklenen girdi | Ayrıntı |
|---|---|
| GeoTIFF ortomozaik (`.tif`) | sıkıştırmasız · LZW (+öngörücü) · Deflate/ZIP · PackBits · JPEG-in-TIFF · şeritli/karolu · 8/16 bit · BigTIFF; GSD, EPSG ve UTM koordinatları otomatik okunur |
| DSM/DTM (`.tif`) | 32-bit float + GDAL_NODATA maskesi |
| `.jpg` / `.png` | coğrafi bilgi olmadan (alanlar piksel cinsinden) |

**Nasıl çalışır?** (5 adım, uygulama içinde yönlendirmeli)

1. **Veri Yükle** — büyük dosyalar seçilen işleme çözünürlüğüne akıllıca küçültülür
   (şerit/karo bazında örnekleme; tüm görüntü belleğe alınmaz).
2. **Etiketle** — 🌲 Çam / Çam değil (+ ek sınıflar) için dört araç: **🪄 Sihirli Değnek**
   (tek tıkla benzer renkli bitişik alanı etiketler, tolerans ayarlı), **⬠ Çokgen**
   (köşe tıkla, çift tık/Enter ile kapat), **▭ Kutu** ve **🖌️ Fırça**;
   tümü geri alınabilir ve proje dosyasına kaydedilir.
3. **Eğit (Oto-AI)** — 24 öznitelik (RGB/HSV istatistikleri, ExG, GLI, VARI, NGRDI,
   doku) çıkarılır; **k-NN, Softmaks Regresyon, Yapay Sinir Ağı ve Rastgele Orman**
   aileleri 3 öznitelik kümesiyle birlikte **33 kombinasyon** halinde katmanlı çapraz
   doğrulamada yarıştırılır ve **en iyi öğrenme şekli otomatik seçilir** (ölçüt: makro-F1;
   tohum=42 ile tekrarlanabilir). Verinizdeki en ayırt edici ortak özellikler Fisher
   skoruyla raporlanır.
   **🔎 Analiz Et:** model hazırken bu bölüme **yeni bir harita/görüntü** bırakın —
   eğitilmiş model korunur, görüntü otomatik sınıflandırılır, karar verilir ve ağaçlar
   nokta atışı işaretlenir (tepe konumları olasılık ağırlıklı merkezle inceltilir).
4. **Sınıflandır** — kayan pencere tüm görüntüyü tarar; güven eşiği ve saydamlık
   kaydırıcılarıyla katmanı ayarlayın; sınıf başına blok/%, **m² alan** (GeoTIFF ise)
   ve bölge sayısı hesaplanır. Üstte **görüntü kararı** verilir: *"çam VAR — %X kaplama"*
   ya da *"çam tespit edilmedi"*.
5. **Çamları işaretle** — ortalama taç çapını girin; çam olasılık yüzeyindeki
   **yerel maksimumlar** NMS ile seyreltilerek **her çam ağacı tek tek işaretlenir ve
   sayılır**; çamların **en yoğun olduğu bölge** ⭐ ile (UTM koordinatıyla) gösterilir.
6. **Dışa aktar** — PNG, CSV, **GeoJSON (QGIS'te açılır)**, **ağaç noktaları GeoJSON**
   (her ağacın UTM koordinatı), model `.json` (başka uçuşta yeniden eğitmeden kullanın),
   proje `.json` (etiketler dahil) ve tez için otomatik **yöntem raporu**.

Çekirdek (TIFF çözücü + ML motoru) `node test/cekirdek_test.mjs` ile test edilir (46 test).
Bu araç, ana YOLOv8-seg hattının **ön etüdü/karşılaştırması** olarak tasarlanmıştır;
ağaç bazlı doğruluk için aşağıdaki pipeline'ı kullanın.

---

## 📦 Kurulum (Windows 11, Node.js 18+)

1. [Node.js LTS](https://nodejs.org) kurun (18 veya üzeri).
2. Proje klasöründe bağımlılıkları kurun:

```bash
npm install
```

Bu komut şu paketleri kurar (Python kütüphanelerinin karşılıkları):

| npm paketi | Karşıladığı Python kütüphanesi | Görev |
|---|---|---|
| `gdal-async` | rasterio, geopandas, shapely, pyproj | GeoTIFF okuma, reproject, geometri işlemleri, Shapefile yazma |
| `onnxruntime-node` | ultralytics (YOLO çalıştırma) | best.onnx modeliyle tahmin |
| `d3-contour` | cv2.findContours | Mask → poligon dönüşümü |
| `munkres-js` | scipy.optimize.linear_sum_assignment | Macar algoritması (eşleştirme) |

> GPU: `onnxruntime-node`, CUDA kuruluysa otomatik `cuda` cihazını dener;
> yoksa `cpu` kullanır ve hangi cihazın kullanıldığını konsola yazar.

### Beklenen klasör yapısı

```
C:\Users\meryemnur.cevik\Desktop\cam_tezi\
├── 02_pix4d_outputs\
│   ├── Proses_transparent_mosaic_group1.tif   (ortomozaik)
│   ├── Proses_dsm.tif                         (DSM, 0.615 cm/px)
│   └── Proses_dtm.tif                         (DTM, 3.08 cm/px)
├── 04_training\
│   └── best.onnx                              (Colab'dan indirilen model, ONNX)
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

### Adım 3 — Modeli ONNX'e Çevirin ve İndirin

Node.js, PyTorch'un `.pt` formatını doğrudan çalıştıramaz; modeli ONNX'e
çevirin. Colab'da eğitimin sonunda **bir kez** şunu çalıştırın:

```python
!yolo export model=cam_segmentation/yolov8m_seg_egitim/weights/best.pt format=onnx imgsz=1280
```

Üretilen `best.onnx` dosyasını bilgisayarınıza, `04_training\best.onnx`
konumuna kopyalayın.

| Girdi | Çıktı |
|---|---|
| `best.pt` | `04_training\best.onnx` |

### Adım 4 — Ortomozaik Üzerinde Tahmin

```bash
node inference_orthomosaic.js
```

Ortomozaik örtüşmeli karolara bölünür, model her karoda tahmin yapar,
tespitler Global NMS ile birleştirilir ve UTM koordinatlı poligonlara dönüştürülür.

| Girdi | Çıktı |
|---|---|
| `best.onnx`, ortomozaik GeoTIFF | `05_inference\cam_tespitleri.geojson` |
| — | `05_inference\tespitler.json` (export için ara dosya) |

### Adım 5 — CBS Çıktıları ve Ağaç Boyu

```bash
node export_results.js
```

DTM, DSM grid'ine **bilinear yöntemle yeniden örneklenir**
(`gdal.reprojectImage`; çözünürlükleri farklıdır: DSM 0.615 cm/px, DTM
3.08 cm/px) ve CHM = DSM − DTM'den her ağacın boyu hesaplanır. Shapefile
öznitelik adları ESRI'nin 10 karakter sınırına uygundur:
`id, conf, alan_m2, cap_m, boy_m, merkez_x, merkez_y`.

| Girdi | Çıktı |
|---|---|
| `tespitler.json`, DSM, DTM | `cam_agaclari.shp` (+ .dbf/.shx/.prj) |
| — | `cam_agaclari.geojson`, `cam_agaclari_ozellikler.csv` |
| — | `ozet_rapor.txt`, `istatistik_grafikleri.html` |

### Adım 6 — Referans Örneklem Kareleri

```bash
node make_reference_grid.js
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
node accuracy_assessment.js
```

Tahmin ve referans poligonları **Macar algoritması** ile birebir eşleştirilir;
**IoU ≥ 0.5** eşleşmeler TP, eşleşmeyen tahminler FP, eşleşmeyen referanslar FN
sayılır. Ağaç bazında Precision / Recall / F1 ve eşleşen çiftler için taç alanı
karşılaştırması (R², RMSE) hesaplanır. **"Model kaç ağacı doğru buldu?"**
sorusunun cevabı bu rapordadır.

| Girdi | Çıktı |
|---|---|
| `cam_agaclari.shp`, `referans_agaclar.shp`, (`grid_kareleri.shp`) | `accuracy_report.txt` |
| — | `alan_karsilastirma_sacilim.html`, `eslesen_ciftler.csv` |

### Adım 9 — Güven Eşiği Duyarlılık Analizi

```bash
node threshold_analysis.js
```

`conf = 0.25, 0.30, 0.40, 0.50, 0.60` eşikleri için tespit sayısı ve toplam
taç alanı hesaplanır (inference **tek sefer** çalışır, eşikler filtrelenir;
fonksiyonlar `inference_orthomosaic.js`'ten import edilir). Sonuç, tezin
**parametre seçimi** bölümüne girer.

| Girdi | Çıktı |
|---|---|
| `best.onnx`, ortomozaik GeoTIFF | `esik_analizi.csv` |
| — | `esik_analizi.html` (eşik–tespit sayısı / toplam alan grafikleri) |

---

## 📄 Proje Dosyaları

| Dosya | Görev |
|---|---|
| `train_yolov8seg.ipynb` | Colab'da YOLOv8m-seg eğitimi (seed=42, deterministic) — tek Python bileşeni |
| `inference_orthomosaic.js` | Ortomozaik üzerinde karo bazlı tahmin (ONNX Runtime) |
| `export_results.js` | Shapefile/GeoJSON/CSV çıktıları + CHM'den ağaç boyu |
| `make_reference_grid.js` | 3 adet 30×30 m referans örneklem karesi (seed=42) |
| `accuracy_assessment.js` | Ağaç bazlı Precision/Recall/F1 + alan karşılaştırması |
| `threshold_analysis.js` | Güven eşiği duyarlılık analizi (CSV + grafik) |
| `lib/ilerleme.js` | İlerleme çubuğu modülü (tqdm karşılığı) |
| `lib/grafik.js` | SVG/HTML grafik modülü (matplotlib karşılığı) |
| `package.json` | Node.js bağımlılıkları (`npm install`) |
| `docs/proje_dokumantasyon.html` | Tüm kodların tek sayfalık HTML dokümantasyonu |

## 💡 Notlar

- Tüm scriptlerin başında **CONFIG** bloğu vardır; yolları oradan değiştirin.
- Tüm çıktı mesajları ve raporlar **Türkçe**dir.
- Grafikler PNG yerine **HTML/SVG** olarak üretilir (`.html` dosyalarını
  tarayıcıda açın; tez için ekran görüntüsü alabilirsiniz). Node.js'te yerel
  bağımlılık olmadan PNG üretimi mümkün olmadığı için bu yöntem seçilmiştir.
- Koordinat sistemi varsayılanı **EPSG:32635** (WGS84 / UTM 35N)'tir;
  farklı bölge için `export_results.js` içindeki `target_crs`'yi güncelleyin.
- Tekrarlanabilirlik: eğitimde `seed=42, deterministic=True`; örneklem
  karelerinde `seed=42` (mulberry32) kullanılır (tezin yöntem bölümünde belirtin).
- Python sürümünden davranış farkları: model dosyası `.pt` yerine `.onnx`,
  ara dosya `tespitler.pkl` yerine `tespitler.json`, grafikler `.png` yerine
  `.html`. İşlem hattının mantığı, eşikleri ve çıktı öznitelikleri birebir aynıdır.
