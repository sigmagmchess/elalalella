#!/usr/bin/env node
// -*- coding: utf-8 -*-
/**
 * YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
 * Referans Veri Hazırlama Yardımcısı (Örneklem Kareleri)
 *
 * Bu script, ortomozaik sınırları içinden RASTGELE 3 adet 30x30 m örneklem
 * karesi seçer ve grid_kareleri.shp olarak kaydeder. Rastgelelik sabit
 * seed (42) ile üretildiği için sonuçlar TEKRARLANABİLİRDİR (tezde belirtin).
 *
 * Kullanıcı bu karelerin içindeki TÜM çam ağaçlarını QGIS'te elle çizerek
 * referans veri setini (referans_agaclar.shp) oluşturur. Bu referans,
 * accuracy_assessment.js ile modelin doğruluğunu ölçmek için kullanılır.
 *
 * NOT (Python → JavaScript çevirisi):
 *     - rasterio / geopandas / shapely → gdal-async
 *     - np.random.default_rng(42) → mulberry32(42) (JavaScript'te yerleşik
 *       seed'li rastgele üreteç olmadığı için deterministik mulberry32
 *       algoritması kullanılır; seed sabit olduğu sürece aynı kareler seçilir)
 *
 * Kullanım:
 *     node make_reference_grid.js
 *
 * Gereksinimler:
 *     - Node.js 18+, npm install (gdal-async)
 *     - Ortomozaik GeoTIFF dosyası
 *
 * Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
 * Tarih: 2024
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import gdal from "gdal-async";

//==============================================================================
// KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
//==============================================================================

export const CONFIG = {
    // Ortomozaik GeoTIFF dosyası (Pix4D çıktısı)
    orthomosaic_path: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\02_pix4d_outputs\Proses_transparent_mosaic_group1.tif`,

    // Çıktı klasörü (grid_kareleri.shp buraya kaydedilecek)
    output_dir: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference`,

    // Örneklem karesi kenar uzunluğu (metre)
    kare_boyutu_m: 30.0,

    // Seçilecek kare sayısı
    kare_sayisi: 3,

    // Rastgelelik tohumu (SABİT: tekrarlanabilirlik için değiştirmeyin)
    seed: 42,

    // Bir karenin kabul edilmesi için izin verilen en fazla boş (NoData) oranı
    // (şeffaf mozaiklerin kenarlarındaki veri içermeyen kareleri elemek için)
    max_bos_oran: 0.5,

    // En fazla deneme sayısı (uygun kare bulunamazsa güvenlik sınırı)
    max_deneme: 2000,
};

//==============================================================================


/**
 * Deterministik rastgele sayı üreteci (mulberry32).
 * Python'daki np.random.default_rng(seed) karşılığı: aynı seed ile her
 * çalıştırmada aynı diziyi üretir (tekrarlanabilirlik).
 *
 * @param {number} seed - Rastgelelik tohumu
 * @returns {() => number} 0-1 aralığında rastgele sayı üreten fonksiyon
 */
function mulberry32(seed) {
    let durum = seed >>> 0;
    return function () {
        durum = (durum + 0x6d2b79f5) >>> 0;
        let t = durum;
        t = Math.imul(t ^ (t >>> 15), t | 1);
        t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
        return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
}


/**
 * Gerekli dosyaların varlığını kontrol eder.
 */
export function dosyaKontrol() {
    console.log("\n📁 Dosya kontrolü yapılıyor...");

    if (!fs.existsSync(CONFIG.orthomosaic_path)) {
        console.log(`❌ Ortomozaik dosyası bulunamadı: ${CONFIG.orthomosaic_path}`);
        console.log("💡 Lütfen CONFIG bölümündeki dosya yolunu kontrol edin.");
        process.exit(1);
    }

    console.log(`   ✅ Ortomozaik: ${CONFIG.orthomosaic_path}`);

    fs.mkdirSync(CONFIG.output_dir, { recursive: true });
    console.log(`   ✅ Çıktı klasörü: ${CONFIG.output_dir}`);
}


/**
 * Bir örneklem karesi içindeki boş (NoData/siyah) piksel oranını hesaplar.
 * Hız için kare, düşük çözünürlükte (64x64) okunur.
 *
 * @param {gdal.Dataset} src - Açık GDAL dataset (ortomozaik)
 * @param {object} kare - {minX, minY, maxX, maxY} (UTM koordinatlı kare)
 * @returns {Promise<number>} 0.0 - 1.0 arasında boş piksel oranı
 */
export async function kareBosOrani(src, kare) {
    try {
        const gt = src.geoTransform;

        // UTM sınırlarını piksel penceresine çevir (kuzeye dönük raster)
        let kolon1 = Math.floor((kare.minX - gt[0]) / gt[1]);
        let kolon2 = Math.ceil((kare.maxX - gt[0]) / gt[1]);
        let satir1 = Math.floor((kare.maxY - gt[3]) / gt[5]);
        let satir2 = Math.ceil((kare.minY - gt[3]) / gt[5]);

        // Raster sınırlarına kırp (rasterio boundless=True, fill=0 karşılığı:
        // kareler zaten tamamen sınır içinde seçildiği için kırpma güvenlidir)
        kolon1 = Math.max(kolon1, 0);
        satir1 = Math.max(satir1, 0);
        kolon2 = Math.min(kolon2, src.rasterSize.x);
        satir2 = Math.min(satir2, src.rasterSize.y);

        const w = kolon2 - kolon1;
        const h = satir2 - satir1;
        if (w <= 0 || h <= 0) return 1.0;

        // Düşük çözünürlükte oku (64x64 yeterli, çok hızlı)
        const veri = await src.bands.get(1).pixels.readAsync(kolon1, satir1, w, h, null, {
            buffer_width: 64,
            buffer_height: 64,
        });

        const nodata = src.bands.get(1).noDataValue;
        let bos = 0;
        for (const v of veri) {
            if ((nodata !== null && v === nodata) || v === 0) bos++;
        }

        return bos / veri.length;
    } catch {
        // Okunamayan kare tamamen boş kabul edilir
        return 1.0;
    }
}


/**
 * Ortomozaik sınırları içinden rastgele, birbiriyle örtüşmeyen ve
 * yeterli veri içeren örneklem kareleri seçer (seed=42, tekrarlanabilir).
 *
 * @param {gdal.Dataset} src - Açık GDAL dataset
 * @returns {Promise<Array>} {minX, minY, maxX, maxY} kareleri
 */
export async function kareleriSec(src) {
    const gt = src.geoTransform;
    const w = src.rasterSize.x;
    const h = src.rasterSize.y;

    // Sınırları hesapla (kuzeye dönük raster varsayımı)
    const bounds = {
        left: gt[0],
        right: gt[0] + w * gt[1],
        top: gt[3],
        bottom: gt[3] + h * gt[5],
    };

    const kare = CONFIG.kare_boyutu_m;

    const genislik = bounds.right - bounds.left;
    const yukseklik = bounds.top - bounds.bottom;

    console.log(`\n🗺️ Ortomozaik kapsamı: ${genislik.toFixed(1)} m x ${yukseklik.toFixed(1)} m`);

    if (genislik < kare || yukseklik < kare) {
        console.log(`❌ HATA: Ortomozaik ${kare.toFixed(0)}x${kare.toFixed(0)} m kare için çok küçük!`);
        process.exit(1);
    }

    // Sabit seed ile rastgele sayı üreteci (tekrarlanabilirlik)
    const rastgele = mulberry32(CONFIG.seed);

    const secilen = [];
    let deneme = 0;

    console.log(
        `\n🎲 Rastgele ${CONFIG.kare_sayisi} adet ${kare.toFixed(0)}x${kare.toFixed(0)} m kare seçiliyor (seed=${CONFIG.seed})...`,
    );

    while (secilen.length < CONFIG.kare_sayisi && deneme < CONFIG.max_deneme) {
        deneme++;

        // Kare tamamen sınırlar içinde kalacak şekilde sol-alt köşe seç
        const x = bounds.left + rastgele() * (genislik - kare);
        const y = bounds.bottom + rastgele() * (yukseklik - kare);

        const aday = { minX: x, minY: y, maxX: x + kare, maxY: y + kare };

        // Daha önce seçilen karelerle örtüşmesin
        const ortusuyor = secilen.some(
            (k) => aday.minX < k.maxX && aday.maxX > k.minX && aday.minY < k.maxY && aday.maxY > k.minY,
        );
        if (ortusuyor) {
            continue;
        }

        // Yeterli veri içersin (kenar boşluklarını ele)
        const bosOran = await kareBosOrani(src, aday);
        if (bosOran > CONFIG.max_bos_oran) {
            continue;
        }

        secilen.push(aday);
        console.log(
            `   ✅ Kare ${secilen.length}: sol-alt köşe (${x.toFixed(1)}, ${y.toFixed(1)}), dolu oran: %${((1 - bosOran) * 100).toFixed(0)}`,
        );
    }

    if (secilen.length < CONFIG.kare_sayisi) {
        console.log(`\n⚠️ UYARI: ${CONFIG.max_deneme} denemede sadece ${secilen.length} uygun kare bulundu.`);
        console.log("   'max_bos_oran' değerini artırmayı deneyin (örn. 0.7).");
        if (!secilen.length) {
            process.exit(1);
        }
    }

    return secilen;
}


/**
 * Ana fonksiyon: kareleri seçer ve shapefile olarak kaydeder.
 */
export async function gridOlustur() {
    console.log("\n" + "=".repeat(60));
    console.log("🌲 REFERANS VERİ HAZIRLAMA - ÖRNEKLEM KARELERİ");
    console.log("=".repeat(60));

    dosyaKontrol();

    const src = await gdal.openAsync(CONFIG.orthomosaic_path);

    if (!src.srs) {
        console.log("❌ HATA: Ortomozaiğin koordinat sistemi (CRS) tanımlı değil!");
        process.exit(1);
    }

    const kod = src.srs.getAuthorityCode(null);
    const ad = src.srs.getAuthorityName(null);
    const crsAdi = kod && ad ? `${ad}:${kod}` : src.srs.toWKT().slice(0, 40) + "...";
    console.log(`   🌍 CRS: ${crsAdi}`);

    const kareler = await kareleriSec(src);
    const srs = src.srs.clone();
    src.close();

    // Shapefile olarak kaydet (GeoDataFrame.to_file karşılığı)
    const shpPath = path.join(CONFIG.output_dir, "grid_kareleri.shp");
    for (const uzanti of [".shp", ".shx", ".dbf", ".prj", ".cpg"]) {
        const p = shpPath.replace(/\.shp$/, uzanti);
        if (fs.existsSync(p)) fs.unlinkSync(p);
    }

    const ds = gdal.drivers.get("ESRI Shapefile").create(shpPath);
    const katman = ds.layers.create("grid_kareleri", srs, gdal.wkbPolygon);
    katman.fields.add(new gdal.FieldDefn("kare_id", gdal.OFTInteger));
    katman.fields.add(new gdal.FieldDefn("alan_m2", gdal.OFTReal));

    kareler.forEach((k, i) => {
        // Dikdörtgen poligon oluştur (shapely box karşılığı)
        const geom = gdal.Geometry.fromGeoJson({
            type: "Polygon",
            coordinates: [[
                [k.minX, k.minY],
                [k.maxX, k.minY],
                [k.maxX, k.maxY],
                [k.minX, k.maxY],
                [k.minX, k.minY],
            ]],
        });
        const f = new gdal.Feature(katman);
        f.setGeometry(geom);
        f.fields.set("kare_id", i + 1);
        f.fields.set("alan_m2", geom.getArea());
        katman.features.add(f);
    });
    ds.close();

    console.log(`\n💾 Kaydedildi: ${shpPath}`);

    // Kullanıcıya QGIS talimatları
    console.log("\n" + "=".repeat(60));
    console.log("📋 ŞİMDİ NE YAPMALISINIZ? (QGIS'TE REFERANS ÇİZİMİ)");
    console.log("=".repeat(60));
    console.log(`
1. QGIS'i açın ve şu katmanları ekleyin:
   - Ortomozaik GeoTIFF  (Layer → Add Layer → Add Raster Layer)
   - grid_kareleri.shp   (Layer → Add Layer → Add Vector Layer)

2. Yeni bir poligon shapefile oluşturun:
   - Layer → Create Layer → New Shapefile Layer
   - Dosya adı : referans_agaclar.shp
   - Konum     : ${CONFIG.output_dir}
   - Geometri  : Polygon
   - CRS       : Ortomozaikle AYNI olmalı (${crsAdi})

3. Her örneklem karesinin İÇİNDEKİ TÜM çam ağaçlarının taçlarını
   tek tek poligon olarak çizin (Toggle Editing → Add Polygon Feature).
   - Taç sınırını olabildiğince gerçeğe yakın çizin.
   - Kare sınırında kalan ağaçlarda, taç merkezi kare İÇİNDEYSE çizin.
   - Hiçbir çamı atlamayın; referans TAM olmalı.

4. Düzenlemeyi kaydedin (Toggle Editing → Save).

5. Ardından doğruluk değerlendirmesini çalıştırın:
   node accuracy_assessment.js
`);
    console.log("=".repeat(60));
    console.log(`\n✅ Tamamlandı! ${kareler.length} örneklem karesi oluşturuldu.`);
    console.log(`   (seed=${CONFIG.seed} sabit olduğu için script yeniden çalıştırılırsa aynı kareler seçilir)`);
}


// Python'daki `if __name__ == "__main__":` bloğunun karşılığı
const anaModulMu =
    process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (anaModulMu) {
    try {
        await gridOlustur();
    } catch (e) {
        console.log(`\n❌ HATA: ${e.message}`);
        console.error(e);
        process.exit(1);
    }
}
