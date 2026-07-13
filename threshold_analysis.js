#!/usr/bin/env node
// -*- coding: utf-8 -*-
/**
 * YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
 * Güven Eşiği Duyarlılık Analizi (Threshold Analysis)
 *
 * Bu script farklı güven eşiği (confidence threshold) değerleri için tespit
 * sayısını ve toplam taç alanını hesaplar. Sonuçlar tezin "parametre seçimi"
 * bölümünde kullanılır.
 *
 * Verimlilik notu: Ortomozaik üzerinde inference SADECE BİR KEZ, en düşük
 * eşikle (örn. 0.25) çalıştırılır. Model conf=0.25 ile çalıştırıldığında
 * 0.25 ve üzeri güvenli TÜM tespitleri döndürdüğü için, daha yüksek eşikler
 * bu tespitlerin güven skoruna göre filtrelenmesiyle elde edilir ve her eşik
 * için Global NMS + alan filtresi yeniden uygulanır. Böylece 5 ayrı inference
 * yerine 1 inference yeterli olur; kod tekrarını önlemek için tüm fonksiyonlar
 * inference_orthomosaic.js'ten import edilir.
 *
 * Kullanım:
 *     node threshold_analysis.js
 *
 * Gereksinimler:
 *     - inference_orthomosaic.js (aynı klasörde; fonksiyonlar oradan import edilir)
 *     - npm install (gdal-async, onnxruntime-node, d3-contour)
 *
 * Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
 * Tarih: 2024
 */

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";

import gdal from "gdal-async";

// inference_orthomosaic.js'teki fonksiyonları import et (kod tekrarı YOK)
import {
    CONFIG as INFERENCE_CONFIG,
    dosyaKontrol,
    modelYukle,
    ortomozaikBilgisi,
    hamTespitleriTopla,
    globalNmsPoligonlar,
    utmPoligonlaraDonustur,
} from "./inference_orthomosaic.js";

import { cizgiSvg, grafikSayfasiKaydet } from "./lib/grafik.js";

//==============================================================================
// KONFİGÜRASYON - Bu değerleri kendi projenize göre düzenleyin
// (Model ve ortomozaik yolları inference_orthomosaic.js CONFIG'inden alınır)
//==============================================================================

export const CONFIG = {
    // Analiz edilecek güven eşiği değerleri
    esikler: [0.25, 0.30, 0.40, 0.50, 0.60],

    // Çıktı klasörü (CSV ve grafik buraya kaydedilir)
    output_dir: String.raw`C:\Users\meryemnur.cevik\Desktop\cam_tezi\05_inference`,
};

//==============================================================================


/**
 * Eşik-duyarlılık grafiğini çizer: ortak x ekseni (güven eşiği) üzerinde
 * iki ayrı panel (tespit sayısı ve toplam taç alanı). İki büyüklüğün
 * ölçeği farklı olduğu için tek grafikte çift y ekseni KULLANILMAZ.
 */
export function grafikCiz(sonuclar, outputDir) {
    const esikler = sonuclar.map((s) => s.esik);

    const paneller = [
        // Panel 1: Tespit sayısı
        cizgiSvg({
            x: esikler,
            y: sonuclar.map((s) => s.tespit_sayisi),
            baslik: "Tespit Edilen Ağaç Sayısı",
            xEtiket: "Güven eşiği (confidence threshold)",
            yEtiket: "Tespit sayısı (adet)",
            renk: "forestgreen",
            degerBicim: (v) => String(Math.round(v)),
        }),
        // Panel 2: Toplam taç alanı
        cizgiSvg({
            x: esikler,
            y: sonuclar.map((s) => s.toplam_alan_m2),
            baslik: "Toplam Taç Alanı",
            xEtiket: "Güven eşiği (confidence threshold)",
            yEtiket: "Toplam taç alanı (m²)",
            renk: "#3b6ea5",
            degerBicim: (v) => v.toFixed(0),
        }),
    ];

    const grafikPath = path.join(outputDir, "esik_analizi.html");
    grafikSayfasiKaydet(grafikPath, "Güven Eşiği Duyarlılık Analizi", paneller);

    console.log(`   ✅ Grafik: ${grafikPath}`);
}


/**
 * Ana eşik analizi fonksiyonu.
 */
export async function analizCalistir() {
    console.log("\n" + "=".repeat(60));
    console.log("🌲 GÜVEN EŞİĞİ DUYARLILIK ANALİZİ");
    console.log("=".repeat(60));

    const esikler = [...CONFIG.esikler].sort((a, b) => a - b);
    const minEsik = esikler[0];

    console.log(`\n📋 Analiz edilecek eşikler: [${esikler.join(", ")}]`);
    console.log(`   Inference SADECE 1 kez, en düşük eşikle (${minEsik}) çalıştırılacak;`);
    console.log("   diğer eşikler güven skoruna göre filtrelenerek hesaplanacak.");

    // Dosya kontrolü (model + ortomozaik, inference CONFIG'inden)
    dosyaKontrol();

    fs.mkdirSync(CONFIG.output_dir, { recursive: true });

    // Model yükle (cihaz seçimi: CPU varsayılan, CUDA varsa otomatik GPU)
    const { model, device } = await modelYukle(INFERENCE_CONFIG.model_path);

    // Ortomozaiği aç ve ham tespitleri EN DÜŞÜK eşikle topla
    console.log(`\n🗺️ Ortomozaik açılıyor: ${INFERENCE_CONFIG.orthomosaic_path}`);

    const src = await gdal.openAsync(INFERENCE_CONFIG.orthomosaic_path);
    let hamTespitler, info;
    try {
        info = ortomozaikBilgisi(src);

        console.log(`   📐 Boyut: ${info.width} x ${info.height} piksel`);
        console.log(`   🌍 CRS: ${info.crs}`);

        hamTespitler = await hamTespitleriTopla(model, src, info, minEsik, device);
    } finally {
        src.close();
    }

    if (!hamTespitler.length) {
        console.log("\n⚠️ UYARI: Hiç tespit yapılamadı, analiz iptal edildi.");
        console.log("   Model veya ortomozaik yollarını ve eşik değerlerini kontrol edin.");
        return;
    }

    // Her eşik için: filtrele → Global NMS → UTM dönüşümü → say/topla
    const sonuclar = [];

    for (const esik of esikler) {
        console.log("\n" + "-".repeat(60));
        console.log(`📊 Eşik analiz ediliyor: conf = ${esik}`);
        console.log("-".repeat(60));

        // Güven skoruna göre filtrele
        const filtreli = hamTespitler.filter((t) => t.confidence >= esik);
        console.log(`   Eşiği geçen ham tespit: ${filtreli.length}`);

        if (!filtreli.length) {
            sonuclar.push({
                esik,
                tespit_sayisi: 0,
                toplam_alan_m2: 0.0,
                ortalama_alan_m2: 0.0,
            });
            console.log("   ⚠️ Bu eşikte hiç tespit kalmadı.");
            continue;
        }

        // Global NMS (her eşik için yeniden; kalan poligon kümesi değişir)
        const nmsSonrasi = globalNmsPoligonlar(filtreli, INFERENCE_CONFIG.global_nms_iou);

        // UTM'e dönüştür + minimum alan filtresi
        const utmPoligonlar = utmPoligonlaraDonustur(nmsSonrasi, info);

        const alanlar = utmPoligonlar.map((p) => p.alan_m2);
        const toplamAlan = alanlar.reduce((a, b) => a + b, 0);

        sonuclar.push({
            esik,
            tespit_sayisi: utmPoligonlar.length,
            toplam_alan_m2: Number(toplamAlan.toFixed(2)),
            ortalama_alan_m2: Number((alanlar.length ? toplamAlan / alanlar.length : 0).toFixed(2)),
        });
    }

    // Sonuç tablosu
    console.log("\n" + "=".repeat(60));
    console.log("📊 EŞİK ANALİZİ SONUÇ TABLOSU");
    console.log("=".repeat(60));
    console.log(" esik  tespit_sayisi  toplam_alan_m2  ortalama_alan_m2");
    for (const s of sonuclar) {
        console.log(
            `${s.esik.toFixed(2).padStart(5)}` +
            `${String(s.tespit_sayisi).padStart(15)}` +
            `${s.toplam_alan_m2.toFixed(2).padStart(16)}` +
            `${s.ortalama_alan_m2.toFixed(2).padStart(18)}`,
        );
    }
    console.log("=".repeat(60));

    // CSV kaydet (utf-8-sig: Excel'de Türkçe karakterler düzgün görünür)
    const csvPath = path.join(CONFIG.output_dir, "esik_analizi.csv");
    const csv = [
        "esik,tespit_sayisi,toplam_alan_m2,ortalama_alan_m2",
        ...sonuclar.map((s) => `${s.esik},${s.tespit_sayisi},${s.toplam_alan_m2},${s.ortalama_alan_m2}`),
    ].join("\n");
    fs.writeFileSync(csvPath, "\uFEFF" + csv + "\n", "utf-8");
    console.log("\n💾 Sonuçlar kaydediliyor...");
    console.log(`   ✅ CSV: ${csvPath}`);

    // Çizgi grafik
    grafikCiz(sonuclar, CONFIG.output_dir);

    console.log("\n✅ Eşik analizi tamamlandı!");
    console.log(`📂 Çıktı klasörü: ${CONFIG.output_dir}`);
    console.log("\n💡 Tez için yorum: Eşik yükseldikçe tespit sayısı azalır (yanlış");
    console.log("   alarmlar elenir ama bazı gerçek ağaçlar da kaçırılır). Doğruluk");
    console.log("   değerlendirmesindeki F1 skorunu maksimize eden eşiği tercih edin.");
}


// Python'daki `if __name__ == "__main__":` bloğunun karşılığı
const anaModulMu =
    process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url);

if (anaModulMu) {
    try {
        await analizCalistir();
    } catch (e) {
        console.log(`\n❌ HATA: ${e.message}`);
        console.error(e);
        process.exit(1);
    }
}
