#!/usr/bin/env node
// -*- coding: utf-8 -*-
/**
 * YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
 * CBS Çıktıları ve İstatistik Raporu Oluşturma Scripti
 *
 * Bu script inference sonuçlarını CBS formatlarına (Shapefile, GeoJSON) dönüştürür,
 * DSM-DTM farkından ağaç boylarını hesaplar ve özet rapor oluşturur.
 *
 * ÖNEMLİ: DSM (0.615 cm/px) ve DTM (3.08 cm/px) çözünürlükleri FARKLIDIR.
 * CHM hesaplanmadan önce DTM, gdal.reprojectImage (rasterio.warp.reproject
 * karşılığı) ile DSM grid'ine bilinear yöntemle yeniden örneklenir;
 * doğrudan dizi çıkarma yapılmaz.
 *
 * NOT (Python → JavaScript çevirisi):
 *   - rasterio / geopandas → gdal-async
 *   - matplotlib           → lib/grafik.js (grafikler PNG yerine HTML/SVG üretir)
 *   - pickle               → JSON (tespitler.json)
 *   - tqdm                 → lib/ilerleme.js
 *
 * Kullanım:
 *     node export_results.js
 *
 * Gereksinimler:
 *     - inference_orthomosaic.js çıktıları (tespitler.json)
 *     - DSM ve DTM GeoTIFF dosyaları (ağaç boyu hesaplama için)
 *
 * Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
 * Tarih: 2024
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import gdal from "gdal-async";

import { ilerleme } from "./lib/ilerleme.js";
import { histogramSvg, grafikSayfasiKaydet } from "./lib/grafik.js";

//==============================================================================
// KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
//==============================================================================

export const CONFIG = {
    // inference_orthomosaic.js çıktı klasörü (tespitler.json burada)
    input_dir: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference`,

    // DSM dosyası (Pix4D çıktısı, 0.615 cm/px) - Ağaç boyu hesaplama için
    dsm_path: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_dsm.tif`,

    // DTM dosyası (Pix4D çıktısı, 3.08 cm/px) - Ağaç boyu hesaplama için
    dtm_path: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_dtm.tif`,

    // Çıktı klasörü (Shapefile ve raporlar için)
    output_dir: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference`,

    // Koordinat sistemi (WGS84 / UTM Zone 35N)
    target_crs: "EPSG:32635",

    // Boy hesaplama yöntemi: "max" (maksimum), "mean" (ortalama), "percentile_95"
    boy_hesaplama: "max",
};

//==============================================================================


/**
 * Gerekli dosyaların varlığını kontrol eder.
 */
export function dosyaKontrol() {
    console.log("\n📁 Dosya kontrolü yapılıyor...");

    const hatalar = [];
    const bulunanlar = {};

    // JSON dosyası (zorunlu)
    const jsonPath = path.join(CONFIG.input_dir, "tespitler.json");
    if (!fs.existsSync(jsonPath)) {
        hatalar.push(`❌ Tespit dosyası bulunamadı: ${jsonPath}`);
        hatalar.push("   Önce node inference_orthomosaic.js çalıştırın!");
    } else {
        console.log(`   ✅ Tespitler: ${jsonPath}`);
        bulunanlar.json = jsonPath;
    }

    // DSM dosyası (opsiyonel ama önerilen)
    if (fs.existsSync(CONFIG.dsm_path)) {
        console.log(`   ✅ DSM: ${CONFIG.dsm_path}`);
        bulunanlar.dsm = CONFIG.dsm_path;
    } else {
        console.log(`   ⚠️ DSM bulunamadı (ağaç boyu hesaplanmayacak): ${CONFIG.dsm_path}`);
    }

    // DTM dosyası (opsiyonel ama önerilen)
    if (fs.existsSync(CONFIG.dtm_path)) {
        console.log(`   ✅ DTM: ${CONFIG.dtm_path}`);
        bulunanlar.dtm = CONFIG.dtm_path;
    } else {
        console.log(`   ⚠️ DTM bulunamadı (ağaç boyu hesaplanmayacak): ${CONFIG.dtm_path}`);
    }

    if (hatalar.length) {
        console.log("\n" + hatalar.join("\n"));
        process.exit(1);
    }

    // Çıktı klasörünü oluştur
    fs.mkdirSync(CONFIG.output_dir, { recursive: true });
    console.log(`   ✅ Çıktı klasörü: ${CONFIG.output_dir}`);

    return bulunanlar;
}


/** NaN'ları atlayarak min/max/ortalama hesaplar (np.nanmin vb. karşılığı). */
function nanIstatistik(dizi) {
    let min = Infinity, max = -Infinity, toplam = 0, sayi = 0;
    for (const v of dizi) {
        if (Number.isNaN(v)) continue;
        if (v < min) min = v;
        if (v > max) max = v;
        toplam += v;
        sayi++;
    }
    return { min, max, ortalama: sayi ? toplam / sayi : NaN, sayi };
}


/**
 * Canopy Height Model (CHM) = DSM - DTM hesaplar.
 * Ağaç boylarını belirlemek için kullanılır.
 *
 * KRİTİK: DSM ve DTM çözünürlükleri farklıdır (DSM ≈ 0.615 cm/px,
 * DTM ≈ 3.08 cm/px). Bu yüzden DTM, gdal.reprojectImage ile DSM'in
 * grid'ine (aynı transform, aynı boyut) bilinear olarak yeniden örneklenir.
 * Doğrudan dizi çıkarma yapılırsa şekil uyuşmazlığı hatası oluşur.
 */
export async function chmHesapla(dsmPath, dtmPath) {
    console.log("\n🌳 CHM (Canopy Height Model) hesaplanıyor...");
    console.log("   CHM = DSM - DTM (ağaç tepesi - zemin = ağaç boyu)");

    try {
        const dsmSrc = await gdal.openAsync(dsmPath);
        const dsmW = dsmSrc.rasterSize.x;
        const dsmH = dsmSrc.rasterSize.y;
        const dsmGt = dsmSrc.geoTransform;
        const dsmSrs = dsmSrc.srs;
        const dsmBand = dsmSrc.bands.get(1);
        const dsmNodata = dsmBand.noDataValue;

        console.log(`   DSM boyut: ${dsmW} x ${dsmH}`);
        console.log(`   DSM çözünürlük: ${(Math.abs(dsmGt[1]) * 100).toFixed(3)} cm/px`);

        const dsmData = new Float32Array(await dsmBand.pixels.readAsync(0, 0, dsmW, dsmH));

        const dtmSrc = await gdal.openAsync(dtmPath);
        console.log(`   DTM orijinal boyut: ${dtmSrc.rasterSize.x} x ${dtmSrc.rasterSize.y}`);
        console.log(`   DTM çözünürlük: ${(Math.abs(dtmSrc.geoTransform[1]) * 100).toFixed(3)} cm/px`);

        // DTM'yi DSM grid'ine yeniden örnekle (bilinear resampling).
        // reprojectImage iki raster'ın geoTransform'larını dikkate aldığı için
        // hem çözünürlük hem de kapsama (extent) farkları doğru işlenir.
        console.log("   🔄 DTM, DSM grid'ine yeniden örnekleniyor (bilinear)...");

        const hedef = gdal.drivers.get("MEM").create("", dsmW, dsmH, 1, gdal.GDT_Float32);
        hedef.geoTransform = dsmGt;
        hedef.srs = dsmSrs;
        hedef.bands.get(1).fill(NaN);
        hedef.bands.get(1).noDataValue = NaN;

        await gdal.reprojectImageAsync({
            src: dtmSrc,
            dst: hedef,
            s_srs: dtmSrc.srs,
            t_srs: dsmSrs,
            resampling: gdal.GRA_Bilinear,
        });

        const dtmResampled = new Float32Array(
            await hedef.bands.get(1).pixels.readAsync(0, 0, dsmW, dsmH),
        );

        dtmSrc.close();
        hedef.close();
        dsmSrc.close();

        // CHM hesapla — artık iki dizi de aynı grid'de, güvenle çıkarılabilir
        const chmData = new Float32Array(dsmW * dsmH);
        const dtmNodataYok = dtmResampled; // reproject çıkışında NoData = NaN
        for (let i = 0; i < chmData.length; i++) {
            const dsmV = dsmData[i];
            // DSM NoData değerlerini NaN yap (DTM NoData'sı reproject'te NaN oldu)
            if (dsmNodata !== null && dsmV === dsmNodata) {
                chmData[i] = NaN;
                continue;
            }
            let v = dsmV - dtmNodataYok[i];
            // Negatif değerleri sıfırla (hatalı veriler)
            if (v < 0) v = 0;
            // Çok yüksek değerleri filtrele (hatalı veriler, >50m mantıksız)
            if (v > 50) v = NaN;
            chmData[i] = v;
        }

        const ist = nanIstatistik(chmData);
        console.log("   ✅ CHM hesaplandı");
        console.log(`   Min boy: ${ist.min.toFixed(2)} m`);
        console.log(`   Max boy: ${ist.max.toFixed(2)} m`);
        console.log(`   Ortalama boy: ${ist.ortalama.toFixed(2)} m`);

        return {
            data: chmData,
            width: dsmW,
            height: dsmH,
            transform: { a: dsmGt[1], b: dsmGt[2], c: dsmGt[0], d: dsmGt[4], e: dsmGt[5], f: dsmGt[3] },
        };
    } catch (e) {
        console.log(`   ❌ CHM hesaplama hatası: ${e.message}`);
        return null;
    }
}


/**
 * Bir noktanın poligon içinde olup olmadığını ışın izleme (ray casting)
 * yöntemiyle belirler; tüm halkalar çift-tek kuralıyla değerlendirilir.
 * (rasterio.features.geometry_mask'ın piksel testi karşılığı)
 */
function noktaPoligonIcinde(x, y, halkalar) {
    let icinde = false;
    for (const halka of halkalar) {
        for (let i = 0, j = halka.length - 1; i < halka.length; j = i++) {
            const [xi, yi] = halka[i];
            const [xj, yj] = halka[j];
            if (yi > y !== yj > y && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) {
                icinde = !icinde;
            }
        }
    }
    return icinde;
}


/** Yüzdelik hesaplar (np.percentile'ın doğrusal interpolasyonu). */
function yuzdelik(degerler, p) {
    const sirali = [...degerler].sort((a, b) => a - b);
    const konum = ((sirali.length - 1) * p) / 100;
    const alt = Math.floor(konum);
    const ust = Math.min(alt + 1, sirali.length - 1);
    return sirali[alt] + (sirali[ust] - sirali[alt]) * (konum - alt);
}


/**
 * Bir poligon içindeki piksellerden ağaç boyunu hesaplar.
 *
 * @param {object} polygonGeoJson - GeoJSON Polygon (koordinatlar UTM)
 * @param {object} chm - chmHesapla() çıktısı
 * @param {string} method - "max", "mean" veya "percentile_95"
 * @returns {number|null} Ağaç boyu (metre) veya null
 */
export function poligonIcinBoyHesapla(polygonGeoJson, chm, method = "max") {
    try {
        const halkalar = polygonGeoJson.coordinates;
        const t = chm.transform;

        // Poligonun UTM sınır kutusunu bul
        let minX = Infinity, minY = Infinity, maxX = -Infinity, maxY = -Infinity;
        for (const [x, y] of halkalar[0]) {
            if (x < minX) minX = x;
            if (x > maxX) maxX = x;
            if (y < minY) minY = y;
            if (y > maxY) maxY = y;
        }

        // UTM → piksel dönüşümü (kuzeye dönük raster: b = d = 0)
        const kolonBas = Math.max(Math.floor((minX - t.c) / t.a), 0);
        const kolonSon = Math.min(Math.ceil((maxX - t.c) / t.a), chm.width - 1);
        // e negatif olduğu için maxY üst satıra karşılık gelir
        const satirBas = Math.max(Math.floor((maxY - t.f) / t.e), 0);
        const satirSon = Math.min(Math.ceil((minY - t.f) / t.e), chm.height - 1);

        // Poligon içindeki değerleri topla (piksel merkezi testi)
        const degerler = [];
        for (let satir = satirBas; satir <= satirSon; satir++) {
            const y = t.f + (satir + 0.5) * t.e;
            for (let kolon = kolonBas; kolon <= kolonSon; kolon++) {
                const x = t.c + (kolon + 0.5) * t.a;
                if (!noktaPoligonIcinde(x, y, halkalar)) continue;
                const v = chm.data[satir * chm.width + kolon];
                if (!Number.isNaN(v)) degerler.push(v);
            }
        }

        if (!degerler.length) {
            return null;
        }

        if (method === "max") {
            return Math.max(...degerler);
        } else if (method === "mean") {
            return degerler.reduce((a, b) => a + b, 0) / degerler.length;
        } else if (method === "percentile_95") {
            return yuzdelik(degerler, 95);
        }
        return Math.max(...degerler);
    } catch {
        return null;
    }
}


/** Basit istatistikler (pandas .mean/.min/.max/.std karşılığı). */
function istatistik(dizi) {
    const n = dizi.length;
    const ortalama = dizi.reduce((a, b) => a + b, 0) / n;
    const varyans = n > 1 ? dizi.reduce((a, b) => a + (b - ortalama) ** 2, 0) / (n - 1) : 0;
    return {
        toplam: dizi.reduce((a, b) => a + b, 0),
        ortalama,
        min: Math.min(...dizi),
        max: Math.max(...dizi),
        std: Math.sqrt(varyans),
    };
}


/**
 * Özet istatistik raporu ve grafikler oluşturur.
 */
export function istatistikRaporuOlustur(kayitlar, crs, outputDir) {
    console.log("\n📊 İstatistik raporu oluşturuluyor...");

    const alanlar = kayitlar.map((k) => k.alan_m2);
    const caplar = kayitlar.map((k) => k.cap_m);
    const guvenler = kayitlar.map((k) => k.conf);
    const boylar = kayitlar.filter((k) => k.boy_m !== null && k.boy_m !== undefined).map((k) => k.boy_m);

    const aIst = istatistik(alanlar);
    const cIst = istatistik(caplar);
    const gIst = istatistik(guvenler);

    // Rapor dosyası
    const raporPath = path.join(outputDir, "ozet_rapor.txt");

    let r = "";
    r += "=".repeat(60) + "\n";
    r += "YTÜ HARİTA MÜHENDİSLİĞİ - ÇAM AĞACI TESPİT RAPORU\n";
    r += "=".repeat(60) + "\n\n";

    r += "GENEL İSTATİSTİKLER\n";
    r += "-".repeat(40) + "\n";
    r += `Toplam tespit edilen ağaç sayısı: ${kayitlar.length}\n`;
    r += `Koordinat sistemi: ${crs}\n\n`;

    r += "TAÇ ALANI İSTATİSTİKLERİ\n";
    r += "-".repeat(40) + "\n";
    r += `Toplam taç alanı: ${aIst.toplam.toFixed(2)} m²\n`;
    r += `Ortalama taç alanı: ${aIst.ortalama.toFixed(2)} m²\n`;
    r += `Minimum taç alanı: ${aIst.min.toFixed(2)} m²\n`;
    r += `Maksimum taç alanı: ${aIst.max.toFixed(2)} m²\n`;
    r += `Std sapma: ${aIst.std.toFixed(2)} m²\n\n`;

    r += "TAÇ ÇAPI İSTATİSTİKLERİ\n";
    r += "-".repeat(40) + "\n";
    r += `Ortalama taç çapı: ${cIst.ortalama.toFixed(2)} m\n`;
    r += `Minimum taç çapı: ${cIst.min.toFixed(2)} m\n`;
    r += `Maksimum taç çapı: ${cIst.max.toFixed(2)} m\n`;
    r += `Std sapma: ${cIst.std.toFixed(2)} m\n\n`;

    if (boylar.length) {
        const bIst = istatistik(boylar);
        r += "AĞAÇ BOYU İSTATİSTİKLERİ\n";
        r += "-".repeat(40) + "\n";
        r += `Boy hesaplanan ağaç sayısı: ${boylar.length}\n`;
        r += `Ortalama ağaç boyu: ${bIst.ortalama.toFixed(2)} m\n`;
        r += `Minimum ağaç boyu: ${bIst.min.toFixed(2)} m\n`;
        r += `Maksimum ağaç boyu: ${bIst.max.toFixed(2)} m\n`;
        r += `Std sapma: ${bIst.std.toFixed(2)} m\n\n`;
    }

    r += "GÜVEN SKORU İSTATİSTİKLERİ\n";
    r += "-".repeat(40) + "\n";
    r += `Ortalama güven: ${gIst.ortalama.toFixed(3)}\n`;
    r += `Minimum güven: ${gIst.min.toFixed(3)}\n`;
    r += `Maksimum güven: ${gIst.max.toFixed(3)}\n\n`;

    r += "=".repeat(60) + "\n";
    r += "Rapor otomatik olarak oluşturulmuştur.\n";
    r += "YTÜ Harita Mühendisliği Yüksek Lisans Tezi\n";
    r += "=".repeat(60) + "\n";

    fs.writeFileSync(raporPath, r, "utf-8");
    console.log(`   ✅ Rapor: ${raporPath}`);

    // Grafikler (matplotlib 2x2 subplot karşılığı: 4 panel, tek HTML sayfası)
    const paneller = [
        // 1. Taç alanı histogramı
        histogramSvg({
            degerler: alanlar, renk: "forestgreen",
            baslik: "Taç Alanı Dağılımı", xEtiket: "Taç Alanı (m²)", yEtiket: "Ağaç Sayısı",
            ortalamaBirim: " m²",
        }),
        // 2. Taç çapı histogramı
        histogramSvg({
            degerler: caplar, renk: "darkgreen",
            baslik: "Taç Çapı Dağılımı", xEtiket: "Taç Çapı (m)", yEtiket: "Ağaç Sayısı",
            ortalamaBirim: " m",
        }),
        // 3. Ağaç boyu histogramı (varsa)
        boylar.length
            ? histogramSvg({
                degerler: boylar, renk: "saddlebrown",
                baslik: "Ağaç Boyu Dağılımı", xEtiket: "Ağaç Boyu (m)", yEtiket: "Ağaç Sayısı",
                ortalamaBirim: " m",
            })
            : `<svg viewBox="0 0 480 380" xmlns="http://www.w3.org/2000/svg"><text x="240" y="185" text-anchor="middle" class="etiket" fill="gray">Ağaç boyu verisi mevcut değil</text><text x="240" y="205" text-anchor="middle" class="etiket" fill="gray">(DSM/DTM gerekli)</text></svg>`,
        // 4. Güven skoru dağılımı
        histogramSvg({
            degerler: guvenler, bolme: 20, renk: "steelblue",
            baslik: "Model Güven Skoru Dağılımı", xEtiket: "Güven Skoru", yEtiket: "Ağaç Sayısı",
            ortalamaBirim: "",
        }),
    ];

    const histogramPath = path.join(outputDir, "istatistik_grafikleri.html");
    grafikSayfasiKaydet(histogramPath, "Çam Ağacı Tespit Analizi - YTÜ Harita Mühendisliği", paneller);

    console.log(`   ✅ Grafikler: ${histogramPath}`);
}


/**
 * Ana export fonksiyonu.
 */
export async function exportCalistir() {
    console.log("\n" + "=".repeat(60));
    console.log("🌲 ÇAM AĞACI CBS ÇIKTILARI VE RAPOR OLUŞTURMA");
    console.log("=".repeat(60));

    // Dosya kontrolü
    const bulunanlar = dosyaKontrol();

    // Tespitleri yükle
    console.log("\n📂 Tespit verileri yükleniyor...");

    const data = JSON.parse(fs.readFileSync(bulunanlar.json, "utf-8"));
    let kayitlar = data.kayitlar;
    console.log(`   ✅ ${kayitlar.length} adet tespit yüklendi`);

    // ESRI Shapefile alan adları en fazla 10 karakter olabilir.
    // "confidence" → "conf" olarak kısaltılır; tüm alan adları 10 karakter
    // sınırına uygundur: id, conf, alan_m2, cap_m, boy_m, merkez_x, merkez_y
    kayitlar = kayitlar.map((k) => ({
        id: k.id,
        conf: k.confidence,
        alan_m2: k.alan_m2,
        cap_m: k.cap_m,
        merkez_x: k.merkez_x,
        merkez_y: k.merkez_y,
        geometry: k.geometry,
    }));

    // CRS kontrolü ve dönüşümü
    let crs = data.crs || null;
    if (!crs) {
        crs = CONFIG.target_crs;
        console.log(`   CRS atandı: ${CONFIG.target_crs}`);
    } else if (crs !== CONFIG.target_crs) {
        // Geometrileri hedef CRS'e dönüştür (gdf.to_crs karşılığı)
        const kaynakSrs = gdal.SpatialReference.fromUserInput(crs);
        const hedefSrs = gdal.SpatialReference.fromUserInput(CONFIG.target_crs);
        const ct = new gdal.CoordinateTransformation(kaynakSrs, hedefSrs);
        kayitlar = kayitlar.map((k) => {
            const g = gdal.Geometry.fromGeoJson(k.geometry);
            g.transform(ct);
            const merkez = g.centroid().toObject().coordinates;
            return { ...k, geometry: g.toObject(), merkez_x: merkez[0], merkez_y: merkez[1] };
        });
        crs = CONFIG.target_crs;
        console.log(`   CRS dönüştürüldü: ${CONFIG.target_crs}`);
    }

    // CHM'den ağaç boyu hesapla
    if (bulunanlar.dsm && bulunanlar.dtm) {
        const chm = await chmHesapla(bulunanlar.dsm, bulunanlar.dtm);

        if (chm !== null) {
            console.log("\n🌳 Her ağaç için boy hesaplanıyor...");

            for (const kayit of ilerleme(kayitlar, "   Boy hesaplama")) {
                kayit.boy_m = poligonIcinBoyHesapla(kayit.geometry, chm, CONFIG.boy_hesaplama);
            }

            const gecerliBoy = kayitlar.filter((k) => k.boy_m !== null).length;
            console.log(`   ✅ ${gecerliBoy}/${kayitlar.length} ağaç için boy hesaplandı`);
        } else {
            kayitlar.forEach((k) => { k.boy_m = null; });
        }
    } else {
        console.log("\n⚠️ DSM/DTM bulunamadı, ağaç boyu hesaplanamıyor");
        kayitlar.forEach((k) => { k.boy_m = null; });
    }

    // Sayısal değerleri yuvarla
    const yuvarla = (v, b) => (v === null || v === undefined ? null : Number(v.toFixed(b)));
    for (const k of kayitlar) {
        k.conf = yuvarla(k.conf, 4);
        k.alan_m2 = yuvarla(k.alan_m2, 3);
        k.cap_m = yuvarla(k.cap_m, 3);
        k.merkez_x = yuvarla(k.merkez_x, 3);
        k.merkez_y = yuvarla(k.merkez_y, 3);
        k.boy_m = yuvarla(k.boy_m, 2);
    }

    // Vektör katmanlarını yaz (Shapefile 10 karakter sınırına uygun adlarla)
    console.log("\n💾 CBS dosyaları kaydediliyor...");

    const srs = gdal.SpatialReference.fromUserInput(crs);
    const alanAdlari = ["id", "conf", "alan_m2", "cap_m", "boy_m", "merkez_x", "merkez_y"];

    const katmanYaz = (surucu, dosyaYolu) => {
        // Var olan dosyaların üzerine yazılabilmesi için önce sil
        if (fs.existsSync(dosyaYolu)) {
            if (surucu === "ESRI Shapefile") {
                for (const uzanti of [".shp", ".shx", ".dbf", ".prj", ".cpg"]) {
                    const p = dosyaYolu.replace(/\.shp$/, uzanti);
                    if (fs.existsSync(p)) fs.unlinkSync(p);
                }
            } else {
                fs.unlinkSync(dosyaYolu);
            }
        }
        const ds = gdal.drivers.get(surucu).create(dosyaYolu);
        const katman = ds.layers.create("cam_agaclari", srs, gdal.wkbPolygon);
        katman.fields.add(new gdal.FieldDefn("id", gdal.OFTInteger));
        for (const ad of alanAdlari.slice(1)) {
            katman.fields.add(new gdal.FieldDefn(ad, gdal.OFTReal));
        }
        for (const k of kayitlar) {
            const f = new gdal.Feature(katman);
            f.setGeometry(gdal.Geometry.fromGeoJson(k.geometry));
            for (const ad of alanAdlari) {
                if (k[ad] !== null && k[ad] !== undefined) f.fields.set(ad, k[ad]);
            }
            katman.features.add(f);
        }
        ds.close();
    };

    // Shapefile olarak kaydet
    const shapefilePath = path.join(CONFIG.output_dir, "cam_agaclari.shp");
    katmanYaz("ESRI Shapefile", shapefilePath);
    console.log(`   ✅ Shapefile: ${shapefilePath}`);

    // GeoJSON olarak kaydet
    const geojsonPath = path.join(CONFIG.output_dir, "cam_agaclari.geojson");
    katmanYaz("GeoJSON", geojsonPath);
    console.log(`   ✅ GeoJSON: ${geojsonPath}`);

    // CSV olarak da kaydet (öznitelik tablosu; utf-8-sig = BOM'lu UTF-8,
    // Excel'de Türkçe karakterler düzgün görünür)
    const csvPath = path.join(CONFIG.output_dir, "cam_agaclari_ozellikler.csv");
    const csvSatirlar = [alanAdlari.join(",")];
    for (const k of kayitlar) {
        csvSatirlar.push(alanAdlari.map((ad) => (k[ad] === null || k[ad] === undefined ? "" : k[ad])).join(","));
    }
    fs.writeFileSync(csvPath, "\uFEFF" + csvSatirlar.join("\n") + "\n", "utf-8");
    console.log(`   ✅ CSV: ${csvPath}`);

    // İstatistik raporu oluştur
    istatistikRaporuOlustur(kayitlar, crs, CONFIG.output_dir);

    // Final özet
    const alanlar = kayitlar.map((k) => k.alan_m2);
    const caplar = kayitlar.map((k) => k.cap_m);
    const boylar = kayitlar.filter((k) => k.boy_m !== null).map((k) => k.boy_m);

    console.log("\n" + "=".repeat(60));
    console.log("📊 SONUÇ ÖZETİ");
    console.log("=".repeat(60));
    console.log(`   Toplam ağaç sayısı: ${kayitlar.length}`);
    console.log(`   Toplam taç alanı: ${alanlar.reduce((a, b) => a + b, 0).toFixed(2)} m²`);
    console.log(`   Ortalama taç çapı: ${(caplar.reduce((a, b) => a + b, 0) / caplar.length).toFixed(2)} m`);
    if (boylar.length) {
        console.log(`   Ortalama ağaç boyu: ${(boylar.reduce((a, b) => a + b, 0) / boylar.length).toFixed(2)} m`);
    }
    console.log("=".repeat(60));

    console.log("\n✅ Tüm çıktılar başarıyla oluşturuldu!");
    console.log(`📂 Çıktı klasörü: ${CONFIG.output_dir}`);

    console.log("\n📋 Oluşturulan dosyalar:");
    console.log("   1. cam_agaclari.shp (+ .dbf, .shx, .prj) - QGIS'te açın");
    console.log("   2. cam_agaclari.geojson - Web haritalar için");
    console.log("   3. cam_agaclari_ozellikler.csv - Excel'de analiz için");
    console.log("   4. ozet_rapor.txt - Metin rapor");
    console.log("   5. istatistik_grafikleri.html - Görsel rapor (tarayıcıda açın)");

    console.log("\n💡 QGIS'te görselleştirme:");
    console.log("   1. QGIS'i açın");
    console.log("   2. Ortomozaik GeoTIFF'i ekleyin (Layer → Add Raster Layer)");
    console.log("   3. cam_agaclari.shp'yi ekleyin (Layer → Add Vector Layer)");
    console.log("   4. Shapefile'ı ortomozaiğin üstüne sürükleyin");
    console.log("   5. Poligonları yeşil renk + şeffaf dolgu yapın");

    console.log("\n💡 Sonraki adım: node make_reference_grid.js ile referans kareleri oluşturun");
}


// Python'daki `if __name__ == "__main__":` bloğunun karşılığı
const anaModulMu =
    process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (anaModulMu) {
    try {
        await exportCalistir();
    } catch (e) {
        console.log(`\n❌ HATA: ${e.message}`);
        console.error(e);
        process.exit(1);
    }
}
