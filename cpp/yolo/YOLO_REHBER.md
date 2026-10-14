# Çam-AI × YOLO rehberi — ortomozaikte çam tespiti, iki yöntemin karşılaştırılması

Bu klasör, tezde **ikinci yöntem** olarak YOLO (Ultralytics YOLO11/YOLOv8) nesne tespitini
kurar. Etiketleme, test ayrımı ve puanlama **Çam-AI uygulamasında (kopru.html)** kalır;
yalnız eğitim ve tahmin Google Colab'da (ücretsiz T4 GPU) yapılır. Böylece MLP tabanlı
Çam-AI modeli ile YOLO, **aynı eğitim çamları ve aynı test çamları** üzerinde, aynı
protokolle (taç poligonu içine ≥ 1 tespit = bulundu) karşılaştırılır.

| Dosya | Ne işe yarar |
|---|---|
| `cam_yolo_colab.ipynb` | Colab defteri: kurulum → veri → eğitim → metrikler → ortomozaikte tahmin → GeoJSON |
| `yolo_tespit_birlestir.py` | Karo tespitlerini ortomozaik koordinatına taşıyan birleştirici (defterin içine gömülüdür; ayrıca komut satırından da çalışır) |
| `YOLO_REHBER.md` | Bu rehber |

## Akış (özet)

```
kopru.html ──📦 YOLO veri seti (ZIP)──▶ Google Drive ──▶ Colab defteri (eğitim + tahmin)
     ▲                                                             │
     └────────── 📥 Dış tespitleri yükle ◀── yolo_tespitler.geojson ┘
```

## 1. Uygulamada veri setini üret

1. **Adım 1:** ortomozaiği yükleyin (işleme çözünürlüğü **8192** önerilir → ~3,7 cm/px; karolar 1024 px ≈ 38 m).
   Pix4D karoları ayrı dosyalarsa hepsini birden seçin, uygulama birleştirir.
2. **Doğruluk Analizi:** 🌿 taç poligonları (`cam_alani_polygon.geojson`), 📍 ağaç noktaları, 🟣 kümeler.
3. **🎓 Eğitim çamları:** eğitimde kullanılacak kimlikleri yazıp **Uygula** (ör. tek numaralar `1,3,5,…,37`;
   kümeler daima test). Çam-AI modelini de aynı kimliklerle eğitmelisiniz ki karşılaştırma adil olsun.
4. **Adım 5 → 📦 YOLO veri seti (ZIP):** onay penceresinde eğitim/test taç sayılarını kontrol edin.
   - Eğitim taçları → YOLO kutusu (taç poligonunun sınır dikdörtgeni).
   - Test taçları ve kümeler, **eğitim karolarında 1 m tamponla düz renge boyanır** (model onları görmez).
   - Tahmin karoları maskesizdir ve tüm ortomozaiği kapsar; `tahmin/karolar.json` coğrafi dönüşümü taşır.
   - "Havuzdaki fotoğrafları da ekle" kutusu: tarayıcı yedeğindeki fotoğraf kopyaları ve Çam çokgenleri
     ek eğitim görüntüsü olur (isteğe bağlı; ortomozaik dışı görünüm katar, ZIP büyür).
5. ZIP'i Google Drive'ınıza yükleyin (`MyDrive/cam_yolo_veri_….zip`).

ZIP içeriği:

```
data.yaml                 YOLO veri tanımı (tek sınıf: cam)
images/train, images/val  1024 px karolar (pozitif karoların %80/%20'si + sınırlı arka plan karosu)
labels/train, labels/val  YOLO etiketleri: "0 cx cy w h" (0–1 normalize)
tahmin/images             tüm ortomozaik karoları (maskesiz)
tahmin/karolar.json       meta (EPSG, ox, oy, sx, sy, karo, bindirme, eğitim kimlikleri) + karo köşeleri
README_tr.txt             özet
```

## 2. Colab'da eğit ve tahmin et

1. [colab.research.google.com](https://colab.research.google.com) → **Dosya → Not defteri yükle** → `cam_yolo_colab.ipynb`.
2. **Çalışma zamanı → Çalışma zamanı türünü değiştir → T4 GPU**.
3. 2. hücrede `ZIP_YOLU` değişkenine Drive yolunu yazın (`/content/drive/MyDrive/cam_yolo_veri_….zip`).
4. Hücreleri sırayla çalıştırın. Varsayılan: `yolo11m.pt`, `imgsz=1024`, `epochs=150`, `patience=30`, `batch=8`,
   COCO ön eğitimli, havadan görüntü artırmaları (`flipud/fliplr/degrees`), tohum 42.
   - *CUDA out of memory* → `BATCH=4` ya da `MODEL="yolo11s.pt"`.
   - Colab oturumu düşerse (12 saat sınırı/boşta kalma) `CALISMA` klasörü silinir; `best.pt` ve sonuç ZIP'i Drive'a kopyalanır, oradan devam edin.
5. Defter sonunda `yolo_sonuclar_cam_yolo11m.zip` iner (Drive'a da kopyalanır). İçinde `yolo_tespitler.geojson`,
   `yolo_kutular.geojson`, `best.pt`, `egitim/results.csv|png`, karışıklık matrisi, PR eğrisi, `args.yaml`, `dogrulama_metrikleri.txt`.

## 3. Uygulamada puanla

1. kopru.html'de aynı ortomozaik + aynı referanslar + aynı 🎓 kimlikler yüklü olsun.
2. **Doğruluk Analizi → 📥 Dış tespitleri yükle → `yolo_tespitler.geojson`.** Adım 4'teki güven kaydırıcısı artık YOLO
   güvenini süzer; tuvalde YOLO noktaları görünür.
3. Eşleşme protokolü **Taç poligonu** → **🔎 Ayar tara** (yalnız güven eşiği taranır, %5–%95) → **Değerlendir** → **📄 Doğruluk raporu (TXT)**.
   Raporda "Tespit kaynağı: DIŞ TESPİT dosyası — yolo_tespitler.geojson" satırı ve eşik taraması tablosu bulunur.
4. Kendi Çam-AI modeliniz için **📍 Ağaçları İşaretle** ile aynı adımları tekrarlayın (dış tespit temizlenir, modelin
   işaretleri döner). İki rapor aynı test çamlarını puanladığı için doğrudan karşılaştırılabilir.

## 4. Tezde raporlama

- **Yöntem:** veri seti üretimi (karo boyu, bindirme, maskeleme), YOLO sürümü/modeli, hiperparametreler (`args.yaml`),
  eğitim süresi/epoch, doğrulama mAP50/F1 (`dogrulama_metrikleri.txt`, `results.png`).
- **Bulgular:** test çamları + kümelerde TP/FP/FN, Precision/Recall/F1 (her iki yöntem için aynı protokol, aynı tablo),
  güven eşiği taraması, konum hatası (taç merkezine en yakın tespit).
- **Tartışma:** referans tezde (Öztürk, 2022, YTÜ) YOLOv4/v5 ile F1 0,53–0,89 bildirilmiştir; yarıçap protokolü (katı)
  ile taç poligonu protokolü (gevşek) sonuçlarını birlikte vermek değerlendirmeyi şeffaflaştırır.
- Parametreler (eşik) referansa göre seçildiği için raporun notu aynen tezde yer almalı; ideal olan eşiği doğrulama alt
  kümesinde seçip bağımsız test alanında raporlamaktır.

## Sık sorunlar

| Belirti | Neden / çözüm |
|---|---|
| "Dosyada görüntü kapsamına düşen nokta yok" | GeoJSON CRS'si ortomozaikle farklı ya da farklı görüntü yüklü. ZIP'i üreten görüntüyü aynı çözünürlükle açın. |
| Colab'da `FileNotFoundError: data.yaml` | 2. hücre çalışmadı ya da ZIP yolu yanlış. |
| Eğitimde mAP50 çok düşük (< 0,3) | Kutular az (eğitim tacı sayısı) → fotoğraf kopyalarını ekleyin, epoch'u artırın, `yolo11s` deneyin; 3. hücredeki örnek karolarda kutuların taçlara oturduğunu doğrulayın. |
| Çok sayıda yanlış alarm | 🔎 Ayar tara ile eşiği yükseltin; `yolo_kutular.geojson`'u QGIS'te açıp hangi nesnelerin karıştığına bakın (başka iğne yapraklılar, gölge). |
| ZIP çok büyük (> 1 GB) | "fotoğrafları da ekle" kutusunu kaldırın ya da işleme çözünürlüğünü 6144 yapın. |

Birleştiriciyi defter dışında çalıştırmak için:

```
python3 yolo_tespit_birlestir.py --veri <açılmış_zip_klasörü> --tahmin karo_tahminleri.json --cikti sonuc/
```
