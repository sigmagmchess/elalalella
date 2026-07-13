#!/usr/bin/env node
// -*- coding: utf-8 -*-
/**
 * YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
 * Ortomozaik Üzerinde Inference (Tahmin) Scripti
 *
 * Bu script büyük GeoTIFF ortomozaiği karolara bölerek YOLOv8-seg modeli ile
 * tahmin yapar ve sonuçları georeferanslı poligon olarak çıkarır.
 *
 * Buradaki fonksiyonlar threshold_analysis.js tarafından da import edilir;
 * bu yüzden ana akış (inferenceCalistir) parametreli fonksiyonlara ayrılmıştır.
 *
 * NOT (Python → JavaScript çevirisi):
 *   - ultralytics YOLO  → onnxruntime-node (model best.onnx olarak dışa
 *     aktarılmalıdır; bir kez Colab'da çalıştırın:
 *       yolo export model=best.pt format=onnx imgsz=1280
 *     Üretilen best.onnx dosyasını 04_training klasörüne koyun.)
 *   - rasterio          → gdal-async
 *   - shapely           → gdal-async (OGR geometri işlemleri)
 *   - cv2.findContours  → d3-contour
 *   - tqdm              → lib/ilerleme.js
 *
 * Kullanım:
 *     node inference_orthomosaic.js
 *
 * Gereksinimler:
 *     - Node.js 18+
 *     - npm install (gdal-async, onnxruntime-node, d3-contour)
 *     - Eğitilmiş best.onnx model dosyası
 *     - Ortomozaik GeoTIFF dosyası
 *
 * Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
 * Tarih: 2024
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import gdal from "gdal-async";
import ort from "onnxruntime-node";
import { contours as d3Contours } from "d3-contour";

import { IlerlemeCubugu, ilerleme } from "./lib/ilerleme.js";

//==============================================================================
// KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
//==============================================================================

export const CONFIG = {
    // Model dosyası (Colab'dan indirdiğiniz best.pt'nin ONNX'e çevrilmiş hali;
    // çevirmek için Colab'da: yolo export model=best.pt format=onnx imgsz=1280)
    model_path: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\04_training\best.onnx`,

    // Ortomozaik GeoTIFF dosyası (Pix4D çıktısı)
    orthomosaic_path: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_transparent_mosaic_group1.tif`,

    // Çıktı klasörü (oluşturulacak)
    output_dir: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference`,

    // Karo boyutu (piksel) - GPU belleğine göre ayarlayın
    // (best.onnx bu boyutla dışa aktarılmış olmalıdır: imgsz=1280)
    tile_size: 1280,

    // Örtüşme oranı (%20 = 0.2)
    overlap_ratio: 0.2,

    // Tahmin güven eşiği (0.0-1.0)
    confidence_threshold: 0.4,

    // NMS IoU eşiği (0.0-1.0)
    iou_threshold: 0.5,

    // Global NMS IoU eşiği (karo birleştirme için)
    global_nms_iou: 0.5,

    // Minimum alan filtresi (m²) - çok küçük tespitleri ele
    min_area_m2: 1.0,

    // Cihaz: "cpu" varsayılan; model yüklenirken CUDA varsa otomatik "cuda" seçilir
    device: "cpu",
};

//==============================================================================


/**
 * Gerekli dosyaların varlığını kontrol eder.
 */
export function dosyaKontrol() {
    console.log("\n📁 Dosya kontrolü yapılıyor...");

    const hatalar = [];

    if (!fs.existsSync(CONFIG.model_path)) {
        hatalar.push(`❌ Model dosyası bulunamadı: ${CONFIG.model_path}`);
        hatalar.push("   (best.pt'yi ONNX'e çevirdiniz mi? Colab'da: yolo export model=best.pt format=onnx imgsz=1280)");
    } else {
        console.log(`   ✅ Model: ${CONFIG.model_path}`);
    }

    if (!fs.existsSync(CONFIG.orthomosaic_path)) {
        hatalar.push(`❌ Ortomozaik dosyası bulunamadı: ${CONFIG.orthomosaic_path}`);
    } else {
        const boyutGb = fs.statSync(CONFIG.orthomosaic_path).size / 1024 ** 3;
        console.log(`   ✅ Ortomozaik: ${CONFIG.orthomosaic_path} (${boyutGb.toFixed(2)} GB)`);
    }

    if (hatalar.length) {
        console.log("\n" + hatalar.join("\n"));
        console.log("\n💡 Lütfen CONFIG bölümündeki dosya yollarını kontrol edin.");
        process.exit(1);
    }

    // Çıktı klasörünü oluştur
    fs.mkdirSync(CONFIG.output_dir, { recursive: true });
    console.log(`   ✅ Çıktı klasörü: ${CONFIG.output_dir}`);
}


/**
 * ONNX modelini yükler ve kullanılan cihazı belirler.
 *
 * Python sürümündeki cihaz_sec() + model_yukle() burada birleşiktir, çünkü
 * ONNX Runtime'da cihaz ancak oturum (session) açılırken belirlenebilir:
 * önce CUDA denenir, kullanılamıyorsa CPU'ya düşülür ve hangi cihazın
 * kullanıldığı konsola yazılır.
 *
 * @param {string} modelPath - best.onnx dosya yolu
 * @returns {Promise<{model: ort.InferenceSession, device: string}>}
 */
export async function modelYukle(modelPath) {
    console.log(`\n📦 Model yükleniyor: ${modelPath}`);

    let model, device;
    try {
        // Önce CUDA destekli GPU dene
        model = await ort.InferenceSession.create(modelPath, { executionProviders: ["cuda"] });
        device = "cuda";
        console.log("🖥️ Cihaz: CUDA GPU bulundu → cuda");
    } catch {
        // GPU yoksa CPU'ya düş (CONFIG'deki varsayılan)
        model = await ort.InferenceSession.create(modelPath, { executionProviders: ["cpu"] });
        device = "cpu";
        console.log("🖥️ Cihaz: GPU bulunamadı, CPU kullanılacak (işlem daha yavaş olabilir)");
    }

    CONFIG.device = device;
    console.log(`   ✅ Model yüklendi (device: ${device})`);
    return { model, device };
}


/**
 * Ortomozaik hakkında bilgi döndürür.
 *
 * transform: rasterio Affine ile aynı adlandırma kullanılır:
 *   x = a*col + b*row + c ;  y = d*col + e*row + f
 * (GDAL geoTransform'dan dönüştürülür: a=gt[1], b=gt[2], c=gt[0],
 *  d=gt[4], e=gt[5], f=gt[3])
 */
export function ortomozaikBilgisi(src) {
    const gt = src.geoTransform;
    const transform = { a: gt[1], b: gt[2], c: gt[0], d: gt[4], e: gt[5], f: gt[3] };

    const w = src.rasterSize.x;
    const h = src.rasterSize.y;

    // Köşe koordinatlarından sınırları hesapla
    const koseler = [
        [0, 0], [w, 0], [0, h], [w, h],
    ].map(([col, row]) => [
        transform.a * col + transform.b * row + transform.c,
        transform.d * col + transform.e * row + transform.f,
    ]);
    const bounds = {
        left: Math.min(...koseler.map((k) => k[0])),
        right: Math.max(...koseler.map((k) => k[0])),
        bottom: Math.min(...koseler.map((k) => k[1])),
        top: Math.max(...koseler.map((k) => k[1])),
    };

    // CRS'i "EPSG:xxxxx" biçiminde yaz (rasterio str(src.crs) karşılığı)
    let crs = "";
    if (src.srs) {
        const kod = src.srs.getAuthorityCode(null);
        const ad = src.srs.getAuthorityName(null);
        crs = kod && ad ? `${ad}:${kod}` : src.srs.toWKT();
    }

    return {
        width: w,
        height: h,
        bands: src.bands.count(),
        crs,
        srs: src.srs,
        transform,
        bounds,
        pixel_size_x: Math.abs(transform.a),
        pixel_size_y: Math.abs(transform.e),
    };
}


/**
 * Örtüşmeli karo pencereleri oluşturur.
 *
 * @returns {Array<[number, number, number, number]>} (x_offset, y_offset, w, h) listesi
 */
export function karoPencereleriniOlustur(width, height, tileSize, overlap) {
    const stride = Math.floor(tileSize * (1 - overlap));
    const pencereler = [];

    let y = 0;
    while (y < height) {
        let x = 0;
        while (x < width) {
            // Karo boyutunu sınırla (kenarlar için)
            const w = Math.min(tileSize, width - x);
            const h = Math.min(tileSize, height - y);

            // Çok küçük karoları atla
            if (w >= Math.floor(tileSize / 4) && h >= Math.floor(tileSize / 4)) {
                pencereler.push([x, y, w, h]);
            }

            x += stride;
        }
        y += stride;
    }

    return pencereler;
}


/**
 * Piksel koordinatlarını UTM koordinatlarına dönüştürür.
 * rasterio.transform.xy'nin varsayılanı gibi piksel MERKEZİ (+0.5) kullanılır.
 *
 * @param {Array<[number, number]>} pikselCoords - (col, row) piksel koordinatları
 * @param {object} transform - Affine dönüşüm (a..f)
 * @returns {Array<[number, number]>} (x, y) UTM koordinatları
 */
export function pikselToUtm(pikselCoords, transform) {
    return pikselCoords.map(([col, row]) => {
        const c = col + 0.5;
        const r = row + 0.5;
        return [
            transform.a * c + transform.b * r + transform.c,
            transform.d * c + transform.e * r + transform.f,
        ];
    });
}


/** Kapalı halkanın işaretli alanını hesaplar (ayakkabı bağı formülü). */
function halkaAlani(halka) {
    let alan = 0;
    for (let i = 0; i < halka.length - 1; i++) {
        alan += halka[i][0] * halka[i + 1][1] - halka[i + 1][0] * halka[i][1];
    }
    return Math.abs(alan) / 2;
}


/**
 * Binary mask'ı GDAL (OGR) poligonuna dönüştürür.
 * cv2.findContours(RETR_EXTERNAL) karşılığı olarak d3-contour kullanılır ve
 * Python'daki gibi yalnızca EN BÜYÜK dış kontur alınır.
 *
 * @param {Float32Array} mask - w*h uzunluğunda mask dizisi (satır öncelikli)
 * @param {number} w - Mask genişliği
 * @param {number} h - Mask yüksekliği
 * @param {number} offsetX - Karonun piksel offseti (x)
 * @param {number} offsetY - Karonun piksel offseti (y)
 * @returns {gdal.Geometry|null} Poligon veya null
 */
export function maskToPolygon(mask, w, h, offsetX, offsetY) {
    // Kontur bul (0.5 eşiği: mask > 0.5)
    const konturKumesi = d3Contours().size([w, h]).thresholds([0.5])(mask);

    if (!konturKumesi.length || !konturKumesi[0].coordinates.length) {
        return null;
    }

    // En büyük dış konturu al (cv2.contourArea ile en büyüğünü seçme karşılığı)
    let enBuyukHalka = null;
    let enBuyukAlan = 0;
    for (const poligon of konturKumesi[0].coordinates) {
        const disHalka = poligon[0]; // İlk halka dış sınırdır (RETR_EXTERNAL)
        const alan = halkaAlani(disHalka);
        if (alan > enBuyukAlan) {
            enBuyukAlan = alan;
            enBuyukHalka = disHalka;
        }
    }

    if (!enBuyukHalka || enBuyukHalka.length < 3) {
        return null;
    }

    // Offset ekle (global piksel koordinatları)
    const noktalar = enBuyukHalka.map(([x, y]) => [x + offsetX, y + offsetY]);

    try {
        let polygon = gdal.Geometry.fromGeoJson({ type: "Polygon", coordinates: [noktalar] });
        if (polygon.isValid() && polygon.getArea() > 0) {
            return polygon;
        }
        // Geçersiz poligonu düzeltmeyi dene (shapely buffer(0) karşılığı)
        polygon = polygon.buffer(0);
        if (polygon.isValid() && polygon.getArea() > 0) {
            return polygon;
        }
    } catch {
        // yoksay
    }

    return null;
}


/** Sigmoid fonksiyonu. */
function sigmoid(v) {
    return 1 / (1 + Math.exp(-v));
}


/**
 * RGB karoyu YOLO girişi için letterbox'lar (ultralytics ön işleme karşılığı):
 * en-boy oranını koruyarak ölçekler, kalan alanı 114 grisi ile doldurur ve
 * CHW düzeninde 0-1 aralığına normalize eder.
 */
function letterboxHazirla(rgb, w, h, hedef) {
    const olcek = Math.min(hedef / w, hedef / h);
    const yeniW = Math.round(w * olcek);
    const yeniH = Math.round(h * olcek);
    const padX = Math.floor((hedef - yeniW) / 2);
    const padY = Math.floor((hedef - yeniH) / 2);

    const girdi = new Float32Array(3 * hedef * hedef).fill(114 / 255);

    // Bilinear yeniden boyutlandırma ile doldur
    for (let y = 0; y < yeniH; y++) {
        const ky = Math.min((y + 0.5) / olcek - 0.5, h - 1);
        const y0 = Math.max(Math.floor(ky), 0);
        const y1 = Math.min(y0 + 1, h - 1);
        const fy = ky - y0;
        for (let x = 0; x < yeniW; x++) {
            const kx = Math.min((x + 0.5) / olcek - 0.5, w - 1);
            const x0 = Math.max(Math.floor(kx), 0);
            const x1 = Math.min(x0 + 1, w - 1);
            const fx = kx - x0;
            for (let b = 0; b < 3; b++) {
                const v =
                    rgb[(y0 * w + x0) * 3 + b] * (1 - fx) * (1 - fy) +
                    rgb[(y0 * w + x1) * 3 + b] * fx * (1 - fy) +
                    rgb[(y1 * w + x0) * 3 + b] * (1 - fx) * fy +
                    rgb[(y1 * w + x1) * 3 + b] * fx * fy;
                girdi[b * hedef * hedef + (y + padY) * hedef + (x + padX)] = v / 255;
            }
        }
    }

    return { girdi, olcek, padX, padY };
}


/** İki kutu (x1,y1,x2,y2) arasındaki IoU. */
function kutuIou(a, b) {
    const kx1 = Math.max(a[0], b[0]);
    const ky1 = Math.max(a[1], b[1]);
    const kx2 = Math.min(a[2], b[2]);
    const ky2 = Math.min(a[3], b[3]);
    const kesisim = Math.max(0, kx2 - kx1) * Math.max(0, ky2 - ky1);
    const alanA = (a[2] - a[0]) * (a[3] - a[1]);
    const alanB = (b[2] - b[0]) * (b[3] - b[1]);
    const birlesim = alanA + alanB - kesisim;
    return birlesim > 0 ? kesisim / birlesim : 0;
}


/**
 * YOLOv8-seg ONNX çıktısını çözer (ultralytics model.predict'in karşılığı):
 * kutu + güven filtresi, NMS ve proto katsayılarından instance mask üretimi.
 *
 * @returns {Array<{mask: Float32Array, conf: number}>} Karo boyutunda (w*h) mask'lar
 */
async function yoloSegTahmin(model, rgb, w, h, hedef, confEsigi, iouEsigi) {
    const { girdi, olcek, padX, padY } = letterboxHazirla(rgb, w, h, hedef);

    const girdiTensor = new ort.Tensor("float32", girdi, [1, 3, hedef, hedef]);
    const sonuc = await model.run({ [model.inputNames[0]]: girdiTensor });

    // output0: [1, 4+nc+32, N] tahminler; output1: [1, 32, mh, mw] proto mask'lar
    const out0 = sonuc[model.outputNames[0]];
    const out1 = sonuc[model.outputNames[1]];
    const [, kanal, N] = out0.dims;
    const sinifSayisi = kanal - 36; // 4 kutu + nc sınıf + 32 mask katsayısı
    const veri = out0.data;
    const [, protoSayisi, mh, mw] = out1.dims;
    const proto = out1.data;

    // 1) Güven eşiğini geçen adayları topla
    const adaylar = [];
    for (let i = 0; i < N; i++) {
        // Sınıf skorlarının en büyüğü (tek sınıf: cam)
        let conf = 0;
        for (let s = 0; s < sinifSayisi; s++) {
            const v = veri[(4 + s) * N + i];
            if (v > conf) conf = v;
        }
        if (conf < confEsigi) continue;

        const cx = veri[0 * N + i];
        const cy = veri[1 * N + i];
        const bw = veri[2 * N + i];
        const bh = veri[3 * N + i];
        const kutu = [cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2];

        const katsayilar = new Float32Array(32);
        for (let k = 0; k < 32; k++) {
            katsayilar[k] = veri[(4 + sinifSayisi + k) * N + i];
        }

        adaylar.push({ kutu, conf, katsayilar });
    }

    // 2) NMS (ultralytics'in dahili NMS'inin karşılığı)
    adaylar.sort((a, b) => b.conf - a.conf);
    const secilenler = [];
    for (const aday of adaylar) {
        let ort_ = false;
        for (const s of secilenler) {
            if (kutuIou(aday.kutu, s.kutu) > iouEsigi) { ort_ = true; break; }
        }
        if (!ort_) secilenler.push(aday);
    }

    // 3) Her seçilen tespit için mask üret:
    //    mask(mh x mw) = sigmoid( Σ katsayı_k · proto_k ), kutu ile kırpılır,
    //    letterbox geri alınarak karo boyutuna (w x h) bilinear örneklenir
    //    (Python'daki retina_masks + cv2.resize karşılığı).
    const tespitler = [];
    const protoBoyut = mh * mw;

    for (const t of secilenler) {
        // Proto kombinasyonu (mask grid'inde)
        const kucukMask = new Float32Array(protoBoyut);
        for (let p = 0; p < protoBoyut; p++) {
            let toplam = 0;
            for (let k = 0; k < protoSayisi; k++) {
                toplam += t.katsayilar[k] * proto[k * protoBoyut + p];
            }
            kucukMask[p] = sigmoid(toplam);
        }

        // Kutu sınırları (mask grid ölçeğinde; letterbox uzayı / (hedef/mw))
        const mOran = hedef / mw;
        const mx1 = t.kutu[0] / mOran, my1 = t.kutu[1] / mOran;
        const mx2 = t.kutu[2] / mOran, my2 = t.kutu[3] / mOran;

        // Karo boyutunda mask örnekle
        const mask = new Float32Array(w * h);
        for (let ty = 0; ty < h; ty++) {
            const ly = ty * olcek + padY;      // letterbox y
            const myy = ly / mOran;            // mask grid y
            if (myy < my1 || myy > my2) continue; // kutu dışını sıfır bırak (crop)
            const y0 = Math.max(Math.min(Math.floor(myy - 0.5), mh - 1), 0);
            const y1 = Math.min(y0 + 1, mh - 1);
            const fy = Math.min(Math.max(myy - 0.5 - y0, 0), 1);
            for (let tx = 0; tx < w; tx++) {
                const lx = tx * olcek + padX;
                const mxx = lx / mOran;
                if (mxx < mx1 || mxx > mx2) continue;
                const x0 = Math.max(Math.min(Math.floor(mxx - 0.5), mw - 1), 0);
                const x1 = Math.min(x0 + 1, mw - 1);
                const fx = Math.min(Math.max(mxx - 0.5 - x0, 0), 1);
                mask[ty * w + tx] =
                    kucukMask[y0 * mw + x0] * (1 - fx) * (1 - fy) +
                    kucukMask[y0 * mw + x1] * fx * (1 - fy) +
                    kucukMask[y1 * mw + x0] * (1 - fx) * fy +
                    kucukMask[y1 * mw + x1] * fx * fy;
            }
        }

        tespitler.push({ mask, conf: t.conf });
    }

    return tespitler;
}


/**
 * Ortomozaiği karolara bölerek modelle tarar ve ham tespitleri
 * (piksel koordinatlı poligon + güven skoru) toplar.
 *
 * threshold_analysis.js bu fonksiyonu düşük bir eşikle (örn. 0.25) bir kez
 * çağırıp, daha yüksek eşikleri güven skoruna göre filtreleyerek analiz eder.
 *
 * @param {ort.InferenceSession} model - Yüklenmiş ONNX modeli
 * @param {gdal.Dataset} src - Açık GDAL dataset (ortomozaik)
 * @param {object} info - ortomozaikBilgisi() çıktısı
 * @param {number} confEsigi - Tahmin güven eşiği
 * @param {string} device - "cuda" veya "cpu" (bilgi amaçlı; oturum zaten açık)
 * @param {number|null} tileSize - null ise CONFIG'den alınır
 * @param {number|null} overlapRatio - null ise CONFIG'den alınır
 * @param {number|null} iouThreshold - null ise CONFIG'den alınır
 * @returns {Promise<Array>} [{polygon (piksel), confidence, karo_x, karo_y}, ...]
 */
export async function hamTespitleriTopla(
    model,
    src,
    info,
    confEsigi,
    device,
    tileSize = null,
    overlapRatio = null,
    iouThreshold = null,
) {
    tileSize = tileSize ?? CONFIG.tile_size;
    overlapRatio = overlapRatio ?? CONFIG.overlap_ratio;
    iouThreshold = iouThreshold ?? CONFIG.iou_threshold;

    // Karo pencerelerini oluştur
    const pencereler = karoPencereleriniOlustur(info.width, info.height, tileSize, overlapRatio);

    console.log(`\n📦 Toplam karo sayısı: ${pencereler.length}`);
    console.log(`   Karo boyutu: ${tileSize}px`);
    console.log(`   Örtüşme: %${Math.floor(overlapRatio * 100)}`);
    console.log(`   Güven eşiği: ${confEsigi}`);

    const tumTespitler = [];

    console.log("\n🔍 Inference başlıyor...");

    const cubuk = new IlerlemeCubugu(pencereler.length, "   Karolar işleniyor");

    for (const [xOff, yOff, w, h] of pencereler) {
        try {
            // Karoyu oku (sadece RGB bantları)
            const bantVerileri = [];
            if (info.bands >= 3) {
                for (const b of [1, 2, 3]) {
                    bantVerileri.push(await src.bands.get(b).pixels.readAsync(xOff, yOff, w, h));
                }
            } else {
                const tek = await src.bands.get(1).pixels.readAsync(xOff, yOff, w, h);
                bantVerileri.push(tek, tek, tek);
            }

            // (C, H, W) -> (H, W, C) düzenine getir + NoData kontrolü
            const rgb = new Uint8Array(w * h * 3);
            let hepsiSifir = true;
            let hepsi255 = true;
            for (let p = 0; p < w * h; p++) {
                for (let b = 0; b < 3; b++) {
                    // uint8'e kırp (Python'daki np.clip karşılığı)
                    const v = Math.max(0, Math.min(255, bantVerileri[b][p]));
                    rgb[p * 3 + b] = v;
                    if (v !== 0) hepsiSifir = false;
                    if (v !== 255) hepsi255 = false;
                }
            }

            // Tamamen boş karoları atla
            if (hepsiSifir || hepsi255) {
                cubuk.adim();
                continue;
            }

            // YOLO tahmin (model.predict karşılığı: eşik + NMS + mask üretimi)
            const sonuclar = await yoloSegTahmin(model, rgb, w, h, tileSize, confEsigi, iouThreshold);

            // Her tespit için poligon oluştur
            for (const tespit of sonuclar) {
                const polygon = maskToPolygon(tespit.mask, w, h, xOff, yOff);

                if (polygon !== null) {
                    tumTespitler.push({
                        polygon,
                        confidence: tespit.conf,
                        karo_x: xOff,
                        karo_y: yOff,
                    });
                }
            }
        } catch {
            // Hatalı karoları sessizce atla
        }
        cubuk.adim();
    }
    cubuk.bitir();

    console.log(`\n📊 Ham tespit sayısı: ${tumTespitler.length}`);

    return tumTespitler;
}


/**
 * Tüm karolardaki poligonları birleştirir ve mükerrer tespitleri eler.
 * IoU tabanlı Non-Maximum Suppression uygular.
 *
 * @param {Array} poligonlar - [{polygon, confidence, ...}, ...]
 * @param {number} iouThreshold - IoU eşiği (üstündeki örtüşmeler birleştirilir)
 * @returns {Array} Filtrelenmiş poligon listesi
 */
export function globalNmsPoligonlar(poligonlar, iouThreshold) {
    if (!poligonlar.length) {
        return [];
    }

    console.log(`\n🔄 Global NMS uygulanıyor (${poligonlar.length} poligon)...`);

    // Güven skoruna göre sırala (yüksekten düşüğe)
    poligonlar = [...poligonlar].sort((a, b) => b.confidence - a.confidence);

    const secilen = [];
    const elenenIndexler = new Set();

    const cubuk = new IlerlemeCubugu(poligonlar.length, "   NMS işlemi");

    for (let i = 0; i < poligonlar.length; i++) {
        if (elenenIndexler.has(i)) {
            cubuk.adim();
            continue;
        }

        const p1 = poligonlar[i];
        secilen.push(p1);
        const poly1 = p1.polygon;

        // Bu poligonla örtüşenleri ele
        for (let j = i + 1; j < poligonlar.length; j++) {
            if (elenenIndexler.has(j)) {
                continue;
            }

            const poly2 = poligonlar[j].polygon;

            try {
                // IoU hesapla
                const kesisim = poly1.intersection(poly2).getArea();
                const birlesim = poly1.union(poly2).getArea();

                if (birlesim > 0) {
                    const iou = kesisim / birlesim;
                    if (iou > iouThreshold) {
                        elenenIndexler.add(j);
                    }
                }
            } catch {
                continue;
            }
        }
        cubuk.adim();
    }
    cubuk.bitir();

    console.log(`   ✅ NMS sonrası: ${secilen.length} poligon (elenen: ${elenenIndexler.size})`);

    return secilen;
}


/**
 * Piksel koordinatlı tespit poligonlarını UTM koordinatlarına dönüştürür
 * ve minimum alan filtresini uygular.
 *
 * @param {Array} tespitler - [{polygon (piksel), confidence}, ...]
 * @param {object} info - ortomozaikBilgisi() çıktısı
 * @param {number|null} minAreaM2 - Minimum taç alanı (null ise CONFIG'den)
 * @returns {Array} [{geometry (UTM, GeoJSON), confidence, alan_m2}, ...]
 */
export function utmPoligonlaraDonustur(tespitler, info, minAreaM2 = null) {
    minAreaM2 = minAreaM2 ?? CONFIG.min_area_m2;

    console.log("\n🌍 Koordinat dönüşümü yapılıyor (Piksel → UTM)...");

    const utmPoligonlar = [];

    for (const tespit of ilerleme(tespitler, "   Dönüştürülüyor")) {
        // Poligon köşelerini al (dış halka)
        const coords = tespit.polygon.toObject().coordinates[0];

        // UTM'e dönüştür
        const utmCoords = pikselToUtm(coords, info.transform);

        try {
            const utmPoly = gdal.Geometry.fromGeoJson({ type: "Polygon", coordinates: [utmCoords] });

            if (utmPoly.isValid() && utmPoly.getArea() > 0) {
                // Alan filtresini uygula
                const alanM2 = utmPoly.getArea();

                if (alanM2 >= minAreaM2) {
                    utmPoligonlar.push({
                        geometry: utmPoly,
                        confidence: tespit.confidence,
                        alan_m2: alanM2,
                    });
                }
            }
        } catch {
            continue;
        }
    }

    console.log(`   ✅ Geçerli UTM poligon sayısı: ${utmPoligonlar.length}`);

    return utmPoligonlar;
}


/**
 * Ana inference fonksiyonu.
 */
export async function inferenceCalistir() {
    console.log("\n" + "=".repeat(60));
    console.log("🌲 ÇAM AĞACI INSTANCE SEGMENTATION - ORTOMOZAIK INFERENCE");
    console.log("=".repeat(60));

    // Dosya kontrolü
    dosyaKontrol();

    // Model yükle (cihaz seçimi: CPU varsayılan, CUDA varsa otomatik GPU)
    const { model, device } = await modelYukle(CONFIG.model_path);

    // Ortomozaiği aç
    console.log(`\n🗺️ Ortomozaik açılıyor: ${CONFIG.orthomosaic_path}`);

    const src = await gdal.openAsync(CONFIG.orthomosaic_path);
    try {
        const info = ortomozaikBilgisi(src);

        console.log(`   📐 Boyut: ${info.width} x ${info.height} piksel`);
        console.log(`   🌍 CRS: ${info.crs}`);
        console.log(`   📏 Piksel boyutu: ${(info.pixel_size_x * 100).toFixed(2)} cm/px`);
        console.log(`   🔲 Bant sayısı: ${info.bands}`);

        // Karoları tara ve ham tespitleri topla
        const tumTespitler = await hamTespitleriTopla(
            model,
            src,
            info,
            CONFIG.confidence_threshold,
            device,
        );

        if (!tumTespitler.length) {
            console.log("\n⚠️ UYARI: Hiç tespit yapılamadı!");
            console.log("   Olası nedenler:");
            console.log("   - Güven eşiği çok yüksek (confidence_threshold'u düşürün)");
            console.log("   - Model bu veriye uygun değil");
            console.log("   - Ortomozaikte çam ağacı yok");
            return;
        }

        // Global NMS uygula
        const filtrelenmis = globalNmsPoligonlar(tumTespitler, CONFIG.global_nms_iou);

        // Piksel koordinatlarını UTM'e dönüştür
        const utmPoligonlar = utmPoligonlaraDonustur(filtrelenmis, info);

        if (!utmPoligonlar.length) {
            console.log("\n⚠️ UYARI: UTM dönüşümü sonrası geçerli poligon kalmadı!");
            return;
        }

        // Kayıtları oluştur (GeoDataFrame karşılığı: öznitelik + geometri listesi)
        console.log("\n💾 Sonuçlar kaydediliyor...");

        const kayitlar = utmPoligonlar.map((p, i) => {
            const merkez = p.geometry.centroid().toObject().coordinates;
            return {
                id: i + 1,
                confidence: p.confidence,
                alan_m2: p.alan_m2,
                // Eşdeğer daire çapı
                cap_m: 2 * Math.sqrt(p.alan_m2 / Math.PI),
                // Merkez koordinatları
                merkez_x: merkez[0],
                merkez_y: merkez[1],
                geometry: p.geometry,
            };
        });

        // GeoJSON olarak kaydet (gdf.to_file(driver="GeoJSON") karşılığı)
        const geojsonPath = path.join(CONFIG.output_dir, "cam_tespitleri.geojson");
        if (fs.existsSync(geojsonPath)) fs.unlinkSync(geojsonPath);
        const gjDs = gdal.drivers.get("GeoJSON").create(geojsonPath);
        const gjKatman = gjDs.layers.create("cam_tespitleri", info.srs, gdal.wkbPolygon);
        gjKatman.fields.add(new gdal.FieldDefn("id", gdal.OFTInteger));
        gjKatman.fields.add(new gdal.FieldDefn("confidence", gdal.OFTReal));
        gjKatman.fields.add(new gdal.FieldDefn("alan_m2", gdal.OFTReal));
        gjKatman.fields.add(new gdal.FieldDefn("cap_m", gdal.OFTReal));
        gjKatman.fields.add(new gdal.FieldDefn("merkez_x", gdal.OFTReal));
        gjKatman.fields.add(new gdal.FieldDefn("merkez_y", gdal.OFTReal));
        for (const k of kayitlar) {
            const f = new gdal.Feature(gjKatman);
            f.setGeometry(k.geometry);
            f.fields.set("id", k.id);
            f.fields.set("confidence", k.confidence);
            f.fields.set("alan_m2", k.alan_m2);
            f.fields.set("cap_m", k.cap_m);
            f.fields.set("merkez_x", k.merkez_x);
            f.fields.set("merkez_y", k.merkez_y);
            gjKatman.features.add(f);
        }
        gjDs.close();
        console.log(`   ✅ GeoJSON: ${geojsonPath}`);

        // JSON olarak da kaydet — export_results.js için
        // (Python'daki pickle dosyasının JavaScript karşılığı)
        const jsonPath = path.join(CONFIG.output_dir, "tespitler.json");
        fs.writeFileSync(
            jsonPath,
            JSON.stringify({
                kayitlar: kayitlar.map((k) => ({
                    id: k.id,
                    confidence: k.confidence,
                    alan_m2: k.alan_m2,
                    cap_m: k.cap_m,
                    merkez_x: k.merkez_x,
                    merkez_y: k.merkez_y,
                    geometry: k.geometry.toObject(),
                })),
                crs: info.crs,
                transform: info.transform,
                bounds: info.bounds,
                pixel_size: info.pixel_size_x,
            }),
            "utf-8",
        );
        console.log(`   ✅ JSON: ${jsonPath}`);

        // Özet rapor
        const alanlar = kayitlar.map((k) => k.alan_m2);
        const caplar = kayitlar.map((k) => k.cap_m);
        const guvenler = kayitlar.map((k) => k.confidence);
        const toplam = (d) => d.reduce((a, b) => a + b, 0);

        console.log("\n" + "=".repeat(60));
        console.log("📊 INFERENCE SONUÇ ÖZETİ");
        console.log("=".repeat(60));
        console.log(`   Toplam tespit: ${kayitlar.length}`);
        console.log(`   Toplam taç alanı: ${toplam(alanlar).toFixed(2)} m²`);
        console.log(`   Ortalama taç alanı: ${(toplam(alanlar) / alanlar.length).toFixed(2)} m²`);
        console.log(`   Ortalama taç çapı: ${(toplam(caplar) / caplar.length).toFixed(2)} m`);
        console.log(`   Min güven: ${Math.min(...guvenler).toFixed(3)}`);
        console.log(`   Max güven: ${Math.max(...guvenler).toFixed(3)}`);
        console.log(`   Ortalama güven: ${(toplam(guvenler) / guvenler.length).toFixed(3)}`);
        console.log("=".repeat(60));

        console.log("\n✅ Inference tamamlandı!");
        console.log(`📂 Çıktı klasörü: ${CONFIG.output_dir}`);
        console.log("\n💡 Sonraki adım: node export_results.js çalıştırın");
    } finally {
        src.close();
    }
}


// Python'daki `if __name__ == "__main__":` bloğunun karşılığı
const anaModulMu =
    process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (anaModulMu) {
    try {
        await inferenceCalistir();
    } catch (e) {
        console.log(`\n❌ HATA: ${e.message}`);
        console.error(e);
        process.exit(1);
    }
}
