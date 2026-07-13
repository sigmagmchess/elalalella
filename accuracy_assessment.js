#!/usr/bin/env node
// -*- coding: utf-8 -*-
/**
 * YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
 * Ağaç Bazlı Doğruluk Değerlendirmesi (Accuracy Assessment)
 *
 * Bu script, modelin ürettiği tahmin poligonlarını (cam_agaclari.shp)
 * kullanıcının QGIS'te elle çizdiği referans poligonlarla
 * (referans_agaclar.shp) karşılaştırır ve "model kaç ağacı doğru buldu?"
 * sorusuna cevap verir.
 *
 * Yöntem:
 *     - Tahmin-referans çiftleri Macar algoritması (munkres-js;
 *       scipy.optimize.linear_sum_assignment karşılığı) ile BİREBİR eşleştirilir.
 *     - IoU >= 0.5 olan eşleşmeler True Positive (TP) sayılır.
 *     - Eşleşmeyen tahminler False Positive (FP),
 *       eşleşmeyen referanslar False Negative (FN) olur.
 *     - Precision, Recall ve F1 ağaç bazında hesaplanır.
 *     - Eşleşen çiftler için taç alanı karşılaştırması yapılır
 *       (R², RMSE, saçılım grafiği).
 *
 * NOT (Python → JavaScript çevirisi):
 *     - geopandas / shapely → gdal-async
 *     - scipy.optimize.linear_sum_assignment → munkres-js
 *     - matplotlib → lib/grafik.js (saçılım grafiği HTML/SVG üretir)
 *     - tqdm → lib/ilerleme.js
 *
 * Kullanım:
 *     node accuracy_assessment.js
 *
 * Gereksinimler:
 *     - npm install (gdal-async, munkres-js)
 *     - cam_agaclari.shp (export_results.js çıktısı)
 *     - referans_agaclar.shp (QGIS'te elle çizilen referans)
 *
 * Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
 * Tarih: 2024
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import gdal from "gdal-async";
import munkres from "munkres-js";

import { IlerlemeCubugu } from "./lib/ilerleme.js";
import { sacilimSvg, grafikSayfasiKaydet } from "./lib/grafik.js";

//==============================================================================
// KONFİGÜRASYON - Bu değerleri kendi dosya yollarınıza göre düzenleyin
//==============================================================================

export const CONFIG = {
    // Modelin ürettiği tahmin poligonları (export_results.js çıktısı)
    tahmin_shp: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference\cam_agaclari.shp`,

    // Referans poligonlar (QGIS'te elle çizilen kontrol verisi)
    referans_shp: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference\referans_agaclar.shp`,

    // Örneklem kareleri (make_reference_grid.js çıktısı, OPSİYONEL).
    // Dosya varsa tahminler bu karelerle sınırlandırılır; çünkü referans
    // sadece kareler içinde çizildiği için tüm sahadaki tahminlerle
    // karşılaştırmak Precision'ı haksız yere düşürür.
    grid_shp: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference\grid_kareleri.shp`,

    // Eşleşme için minimum IoU eşiği (tez literatüründe standart: 0.5)
    iou_esigi: 0.5,

    // Çıktı klasörü (rapor ve grafik buraya kaydedilir)
    output_dir: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference`,
};

//==============================================================================


/**
 * Gerekli dosyaların varlığını kontrol eder.
 */
export function dosyaKontrol() {
    console.log("\n📁 Dosya kontrolü yapılıyor...");

    const hatalar = [];

    if (!fs.existsSync(CONFIG.tahmin_shp)) {
        hatalar.push(`❌ Tahmin shapefile bulunamadı: ${CONFIG.tahmin_shp}`);
        hatalar.push("   Önce node inference_orthomosaic.js ve node export_results.js çalıştırın!");
    } else {
        console.log(`   ✅ Tahminler: ${CONFIG.tahmin_shp}`);
    }

    if (!fs.existsSync(CONFIG.referans_shp)) {
        hatalar.push(`❌ Referans shapefile bulunamadı: ${CONFIG.referans_shp}`);
        hatalar.push("   QGIS'te referans poligonları çizip bu yola kaydedin.");
        hatalar.push("   (Yardım için: node make_reference_grid.js)");
    } else {
        console.log(`   ✅ Referans: ${CONFIG.referans_shp}`);
    }

    if (hatalar.length) {
        console.log("\n" + hatalar.join("\n"));
        process.exit(1);
    }

    fs.mkdirSync(CONFIG.output_dir, { recursive: true });
    console.log(`   ✅ Çıktı klasörü: ${CONFIG.output_dir}`);
}


/**
 * Bir shapefile'daki tüm geometrileri okur.
 *
 * @returns {{geometriler: gdal.Geometry[], srs: gdal.SpatialReference}}
 */
function shapefileOku(dosyaYolu) {
    const ds = gdal.open(dosyaYolu);
    const katman = ds.layers.get(0);
    const geometriler = [];
    katman.features.forEach((f) => {
        const g = f.getGeometry();
        if (g) geometriler.push(g.clone());
    });
    const srs = katman.srs;
    return { geometriler, srs, ds };
}


/**
 * Geçersiz geometrileri buffer(0) ile düzeltir, boşları eler.
 */
export function geometriDuzelt(geometriler) {
    const duzgun = [];
    for (let g of geometriler) {
        if (!g) continue;
        if (!g.isValid()) {
            try {
                g = g.buffer(0);
            } catch {
                continue;
            }
        }
        if (g.getArea() > 0) duzgun.push(g);
    }
    return duzgun;
}


/**
 * İki poligon arasındaki IoU (Intersection over Union) değerini hesaplar.
 */
export function iouHesapla(poly1, poly2) {
    try {
        if (!poly1.intersects(poly2)) {
            return 0.0;
        }
        const kesisim = poly1.intersection(poly2).getArea();
        const birlesim = poly1.union(poly2).getArea();
        if (birlesim <= 0) {
            return 0.0;
        }
        return kesisim / birlesim;
    } catch {
        return 0.0;
    }
}


/**
 * Tahmin ve referans poligonlarını Macar algoritması ile BİREBİR eşleştirir.
 *
 * @param {gdal.Geometry[]} tahminler - Tahmin geometrileri
 * @param {gdal.Geometry[]} referanslar - Referans geometrileri
 * @param {number} iouEsigi - TP sayılmak için minimum IoU
 * @returns {{eslesmeler: Array<[number, number, number]>, iouMatrisi: number[][]}}
 *          eslesmeler: (tahmin_index, referans_index, iou) — sadece IoU >= eşik
 */
export function eslestir(tahminler, referanslar, iouEsigi) {
    const nT = tahminler.length;
    const nR = referanslar.length;

    console.log(`\n🔗 Eşleştirme yapılıyor (${nT} tahmin x ${nR} referans)...`);

    // IoU matrisini hesapla
    const iouMatrisi = [];
    const cubuk = new IlerlemeCubugu(nT, "   IoU matrisi");
    for (let i = 0; i < nT; i++) {
        const satir = new Array(nR);
        for (let j = 0; j < nR; j++) {
            satir[j] = iouHesapla(tahminler[i], referanslar[j]);
        }
        iouMatrisi.push(satir);
        cubuk.adim();
    }
    cubuk.bitir();

    // Macar algoritması: toplam IoU'yu MAKSİMİZE eden birebir atama
    // (munkres maliyeti minimize eder, bu yüzden -IoU verilir;
    //  scipy linear_sum_assignment ile aynı yaklaşım)
    const maliyet = iouMatrisi.map((satir) => satir.map((v) => -v));
    const atamalar = munkres(maliyet);

    // Sadece IoU eşiğini geçen atamalar gerçek eşleşme sayılır
    const eslesmeler = [];
    for (const [i, j] of atamalar) {
        // munkres kare olmayan matrisi doldurur; sınır dışı indeksleri atla
        if (i >= nT || j >= nR) continue;
        const iou = iouMatrisi[i][j];
        if (iou >= iouEsigi) {
            eslesmeler.push([i, j, iou]);
        }
    }

    console.log(`   ✅ IoU >= ${iouEsigi} olan eşleşme sayısı: ${eslesmeler.length}`);

    return { eslesmeler, iouMatrisi };
}


/**
 * Eşleşen çiftler için taç alanı karşılaştırması yapar:
 * R², RMSE ve saçılım grafiği (tahmin vs referans).
 *
 * @returns {object|null} {r2, rmse, n, ref_toplam, tah_toplam} veya null
 */
export function alanKarsilastirma(eslesmeler, tahminler, referanslar, outputDir) {
    if (eslesmeler.length < 2) {
        console.log("\n⚠️ Alan karşılaştırması için en az 2 eşleşme gerekli, atlanıyor.");
        return null;
    }

    console.log("\n📐 Taç alanı karşılaştırması yapılıyor (tahmin vs referans)...");

    const refAlan = eslesmeler.map(([, j]) => referanslar[j].getArea());
    const tahAlan = eslesmeler.map(([i]) => tahminler[i].getArea());

    // Doğrusal regresyon ile R² (scipy.stats.linregress karşılığı: r²=Pearson r'nin karesi)
    const n = refAlan.length;
    const ortX = refAlan.reduce((a, b) => a + b, 0) / n;
    const ortY = tahAlan.reduce((a, b) => a + b, 0) / n;
    let sxy = 0, sxx = 0, syy = 0;
    for (let i = 0; i < n; i++) {
        sxy += (refAlan[i] - ortX) * (tahAlan[i] - ortY);
        sxx += (refAlan[i] - ortX) ** 2;
        syy += (tahAlan[i] - ortY) ** 2;
    }
    const r2 = sxx > 0 && syy > 0 ? (sxy * sxy) / (sxx * syy) : 0;

    // RMSE
    const rmse = Math.sqrt(
        refAlan.reduce((a, _, i) => a + (tahAlan[i] - refAlan[i]) ** 2, 0) / n,
    );

    console.log(`   R²   : ${r2.toFixed(4)}`);
    console.log(`   RMSE : ${rmse.toFixed(3)} m²`);

    // Saçılım grafiği (tek seri; 1:1 doğrusu referans çizgisi olarak)
    const svg = sacilimSvg({
        x: refAlan,
        y: tahAlan,
        baslik: "Taç Alanı: Model Tahmini vs Referans",
        xEtiket: "Referans taç alanı (m²)",
        yEtiket: "Tahmin edilen taç alanı (m²)",
        birEBir: true,
        kutu: `n = ${n}\nR² = ${r2.toFixed(3)}\nRMSE = ${rmse.toFixed(2)} m²`,
    });

    const grafikPath = path.join(outputDir, "alan_karsilastirma_sacilim.html");
    grafikSayfasiKaydet(grafikPath, "Taç Alanı: Model Tahmini vs Referans", [svg]);

    console.log(`   ✅ Saçılım grafiği: ${grafikPath}`);

    return {
        r2,
        rmse,
        n,
        ref_toplam: refAlan.reduce((a, b) => a + b, 0),
        tah_toplam: tahAlan.reduce((a, b) => a + b, 0),
    };
}


/**
 * Sonuçları hem konsola hem accuracy_report.txt dosyasına Türkçe yazar.
 */
export function raporYaz(sonuclar, outputDir) {
    const satirlar = [];
    satirlar.push("=".repeat(60));
    satirlar.push("YTÜ HARİTA MÜHENDİSLİĞİ - AĞAÇ BAZLI DOĞRULUK RAPORU");
    satirlar.push("=".repeat(60));
    satirlar.push("");
    satirlar.push("VERİ ÖZETİ");
    satirlar.push("-".repeat(40));
    satirlar.push(`Tahmin poligon sayısı  : ${sonuclar.n_tahmin}`);
    satirlar.push(`Referans poligon sayısı: ${sonuclar.n_referans}`);
    if (sonuclar.grid_kullanildi) {
        satirlar.push("Not: Tahminler örneklem kareleri (grid_kareleri.shp) ile");
        satirlar.push("     sınırlandırılmıştır; referans yalnızca bu karelerde çizilidir.");
    }
    satirlar.push("");
    satirlar.push(`EŞLEŞTİRME SONUÇLARI (IoU eşiği = ${sonuclar.iou_esigi})`);
    satirlar.push("-".repeat(40));
    satirlar.push(`True Positive  (doğru tespit)        : ${sonuclar.tp}`);
    satirlar.push(`False Positive (yanlış alarm)        : ${sonuclar.fp}`);
    satirlar.push(`False Negative (kaçırılan ağaç)      : ${sonuclar.fn}`);
    satirlar.push("");
    satirlar.push("DOĞRULUK METRİKLERİ (AĞAÇ BAZINDA)");
    satirlar.push("-".repeat(40));
    satirlar.push(`Precision (kesinlik)  : ${sonuclar.precision.toFixed(4)}  (%${(sonuclar.precision * 100).toFixed(1)})`);
    satirlar.push(`Recall    (duyarlılık): ${sonuclar.recall.toFixed(4)}  (%${(sonuclar.recall * 100).toFixed(1)})`);
    satirlar.push(`F1 skoru              : ${sonuclar.f1.toFixed(4)}  (%${(sonuclar.f1 * 100).toFixed(1)})`);
    satirlar.push("");
    if (sonuclar.eslesmeler.length) {
        const ortIou =
            sonuclar.eslesmeler.reduce((a, e) => a + e[2], 0) / sonuclar.eslesmeler.length;
        satirlar.push(`Eşleşen çiftlerin ortalama IoU değeri : ${ortIou.toFixed(4)}`);
        satirlar.push("");
    }

    if (sonuclar.alan !== null) {
        const alan = sonuclar.alan;
        satirlar.push("TAÇ ALANI KARŞILAŞTIRMASI (EŞLEŞEN ÇİFTLER)");
        satirlar.push("-".repeat(40));
        satirlar.push(`Karşılaştırılan çift sayısı : ${alan.n}`);
        satirlar.push(`R² (belirleme katsayısı)    : ${alan.r2.toFixed(4)}`);
        satirlar.push(`RMSE                        : ${alan.rmse.toFixed(3)} m²`);
        satirlar.push(`Toplam referans alanı       : ${alan.ref_toplam.toFixed(2)} m²`);
        satirlar.push(`Toplam tahmin alanı         : ${alan.tah_toplam.toFixed(2)} m²`);
        satirlar.push("Saçılım grafiği             : alan_karsilastirma_sacilim.html");
        satirlar.push("");
    }

    satirlar.push("=".repeat(60));
    satirlar.push("Rapor otomatik olarak oluşturulmuştur.");
    satirlar.push("YTÜ Harita Mühendisliği Yüksek Lisans Tezi");
    satirlar.push("=".repeat(60));

    const raporMetni = satirlar.join("\n");

    // Konsola yaz
    console.log("\n" + raporMetni);

    // Dosyaya yaz
    const raporPath = path.join(outputDir, "accuracy_report.txt");
    fs.writeFileSync(raporPath, raporMetni + "\n", "utf-8");

    console.log(`\n✅ Rapor kaydedildi: ${raporPath}`);
}


/**
 * Ana doğruluk değerlendirme fonksiyonu.
 */
export async function degerlendirmeCalistir() {
    console.log("\n" + "=".repeat(60));
    console.log("🌲 AĞAÇ BAZLI DOĞRULUK DEĞERLENDİRMESİ");
    console.log("=".repeat(60));

    dosyaKontrol();

    // Verileri yükle
    console.log("\n📂 Shapefile'lar yükleniyor...");
    const tahminVeri = shapefileOku(CONFIG.tahmin_shp);
    const referansVeri = shapefileOku(CONFIG.referans_shp);

    let tahminler = tahminVeri.geometriler;
    let referanslar = referansVeri.geometriler;

    console.log(`   ✅ Tahmin: ${tahminler.length} poligon`);
    console.log(`   ✅ Referans: ${referanslar.length} poligon`);

    if (!referanslar.length) {
        console.log("\n❌ HATA: Referans shapefile boş! QGIS'te poligon çizdiğinizden emin olun.");
        process.exit(1);
    }

    // CRS uyumu: referansı tahmin CRS'ine dönüştür
    if (
        tahminVeri.srs && referansVeri.srs &&
        !referansVeri.srs.isSame(tahminVeri.srs)
    ) {
        console.log("   🔄 Referans CRS, tahmin CRS'ine dönüştürülüyor...");
        const ct = new gdal.CoordinateTransformation(referansVeri.srs, tahminVeri.srs);
        referanslar = referanslar.map((g) => {
            const kopya = g.clone();
            kopya.transform(ct);
            return kopya;
        });
    }

    // Geometrileri temizle
    tahminler = geometriDuzelt(tahminler);
    referanslar = geometriDuzelt(referanslar);

    // Örneklem kareleri varsa tahminleri karelerle sınırla
    // (referans sadece kareler içinde çizildiği için adil karşılaştırma)
    let gridKullanildi = false;
    if (CONFIG.grid_shp && fs.existsSync(CONFIG.grid_shp)) {
        const gridVeri = shapefileOku(CONFIG.grid_shp);
        let kareler = gridVeri.geometriler;
        if (gridVeri.srs && tahminVeri.srs && !gridVeri.srs.isSame(tahminVeri.srs)) {
            const ct = new gdal.CoordinateTransformation(gridVeri.srs, tahminVeri.srs);
            kareler = kareler.map((g) => {
                const kopya = g.clone();
                kopya.transform(ct);
                return kopya;
            });
        }

        // Kareleri tek geometride birleştir (unary_union karşılığı)
        let karelerBirlesik = kareler[0];
        for (let i = 1; i < kareler.length; i++) {
            karelerBirlesik = karelerBirlesik.union(kareler[i]);
        }

        const onceki = tahminler.length;
        // Taç merkezi kare içinde olan tahminler değerlendirilir
        tahminler = tahminler.filter((g) => g.centroid().within(karelerBirlesik));
        gridKullanildi = true;

        console.log(`\n🔲 Örneklem kareleri bulundu: ${CONFIG.grid_shp}`);
        console.log(`   Tahminler karelerle sınırlandırıldı: ${onceki} → ${tahminler.length}`);
        gridVeri.ds.close();
    } else {
        console.log("\n🔲 Örneklem karesi dosyası yok; TÜM tahminler değerlendirilecek.");
        console.log("   (Referans tüm sahayı kapsamıyorsa Precision olduğundan düşük çıkar!)");
    }

    if (!tahminler.length) {
        console.log("\n❌ HATA: Değerlendirilecek tahmin poligonu kalmadı!");
        console.log("   Grid kareleri ile tahminlerin aynı bölgede olduğundan emin olun.");
        process.exit(1);
    }

    // Macar algoritması ile birebir eşleştir
    const { eslesmeler } = eslestir(tahminler, referanslar, CONFIG.iou_esigi);

    // TP / FP / FN
    const tp = eslesmeler.length;
    const fp = tahminler.length - tp;
    const fn = referanslar.length - tp;

    // Metrikler (sıfıra bölme koruması)
    const precision = tp + fp > 0 ? tp / (tp + fp) : 0.0;
    const recall = tp + fn > 0 ? tp / (tp + fn) : 0.0;
    const f1 = precision + recall > 0 ? (2 * precision * recall) / (precision + recall) : 0.0;

    // Eşleşen çiftleri CSV olarak kaydet (tez eki için)
    if (eslesmeler.length) {
        const basliklar = "tahmin_no,referans_no,iou,tahmin_alan_m2,referans_alan_m2";
        const satirlar = eslesmeler.map(
            ([i, j, iou]) =>
                `${i + 1},${j + 1},${iou.toFixed(4)},${tahminler[i].getArea().toFixed(3)},${referanslar[j].getArea().toFixed(3)}`,
        );
        const eslesmeCsv = path.join(CONFIG.output_dir, "eslesen_ciftler.csv");
        fs.writeFileSync(eslesmeCsv, "\uFEFF" + basliklar + "\n" + satirlar.join("\n") + "\n", "utf-8");
        console.log(`   ✅ Eşleşen çiftler: ${eslesmeCsv}`);
    }

    // Taç alanı karşılaştırması (R², RMSE, saçılım grafiği)
    const alanSonuc = alanKarsilastirma(eslesmeler, tahminler, referanslar, CONFIG.output_dir);

    // Raporu yaz (konsol + accuracy_report.txt)
    raporYaz(
        {
            n_tahmin: tahminler.length,
            n_referans: referanslar.length,
            iou_esigi: CONFIG.iou_esigi,
            tp,
            fp,
            fn,
            precision,
            recall,
            f1,
            eslesmeler,
            alan: alanSonuc,
            grid_kullanildi: gridKullanildi,
        },
        CONFIG.output_dir,
    );

    tahminVeri.ds.close();
    referansVeri.ds.close();

    console.log("\n💡 Sonraki adım: node threshold_analysis.js ile güven eşiği analizini yapın");
}


// Python'daki `if __name__ == "__main__":` bloğunun karşılığı
const anaModulMu =
    process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (anaModulMu) {
    try {
        await degerlendirmeCalistir();
    } catch (e) {
        console.log(`\n❌ HATA: ${e.message}`);
        console.error(e);
        process.exit(1);
    }
}
