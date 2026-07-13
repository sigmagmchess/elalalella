// -*- coding: utf-8 -*-
/**
 * YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
 * Grafik Modülü (Python'daki matplotlib kütüphanesinin karşılığı)
 *
 * Python sürümü grafikleri matplotlib ile PNG olarak üretiyordu. Node.js'te
 * yerel (native) bağımlılık gerektirmeden PNG üretmek mümkün olmadığı için
 * bu modül aynı grafikleri SVG olarak çizer ve kendi kendine yeten bir HTML
 * dosyasına kaydeder. HTML dosyası her tarayıcıda açılır; tezde kullanmak
 * için tarayıcıdan PNG olarak dışa aktarılabilir (sağ tık → görüntüyü kaydet
 * veya ekran görüntüsü).
 *
 * Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
 * Tarih: 2024
 */

import fs from "node:fs";

// Panel boyutları (matplotlib figsize karşılığı)
const GENISLIK = 480;
const YUKSEKLIK = 380;
const KENAR = { ust: 42, sag: 16, alt: 48, sol: 62 };

/** Sayıyı okunur biçimde yazar (eksen etiketleri için). */
function sayiYaz(v) {
    if (!isFinite(v)) return "";
    if (Math.abs(v) >= 1000) return v.toLocaleString("tr-TR", { maximumFractionDigits: 0 });
    if (Number.isInteger(v)) return String(v);
    return v.toFixed(Math.abs(v) < 1 ? 2 : 1);
}

/** Verilen aralık için "güzel" eksen sınırları ve adım üretir. */
function eksenAdimlari(min, max, adet = 5, tamsayi = false) {
    if (min === max) { min -= 1; max += 1; }
    const ham = (max - min) / adet;
    const buyukluk = Math.pow(10, Math.floor(Math.log10(ham)));
    const norm = ham / buyukluk;
    let adim = (norm <= 1 ? 1 : norm <= 2 ? 2 : norm <= 5 ? 5 : 10) * buyukluk;
    // Sayım eksenlerinde kesirli adım anlamsızdır (örn. 0.2 ağaç)
    if (tamsayi && adim < 1) adim = 1;
    const bas = Math.floor(min / adim) * adim;
    const son = Math.ceil(max / adim) * adim;
    const cizgiler = [];
    for (let v = bas; v <= son + adim / 2; v += adim) cizgiler.push(v);
    return { min: bas, max: son, cizgiler };
}

function xmlKacis(s) {
    return String(s).replace(/&/g, "&amp;").replace(/</g, "&lt;").replace(/>/g, "&gt;").replace(/"/g, "&quot;");
}

/** SVG panel iskeleti: başlık, eksenler, ızgara. Çizim alanı bilgisi döndürür. */
function panelBaslat(baslik, xEksen, yEksen) {
    const icW = GENISLIK - KENAR.sol - KENAR.sag;
    const icH = YUKSEKLIK - KENAR.ust - KENAR.alt;
    const xOlcek = (v) => KENAR.sol + ((v - xEksen.min) / (xEksen.max - xEksen.min)) * icW;
    const yOlcek = (v) => KENAR.ust + icH - ((v - yEksen.min) / (yEksen.max - yEksen.min)) * icH;

    let s = "";
    // Izgara (arka planda, soluk — matplotlib grid karşılığı)
    for (const v of yEksen.cizgiler) {
        const y = yOlcek(v);
        s += `<line x1="${KENAR.sol}" y1="${y}" x2="${GENISLIK - KENAR.sag}" y2="${y}" stroke="#d9d9d9" stroke-width="0.6"/>`;
        s += `<text x="${KENAR.sol - 7}" y="${y + 4}" text-anchor="end" class="tik">${sayiYaz(v)}</text>`;
    }
    for (const v of xEksen.cizgiler) {
        const x = xOlcek(v);
        s += `<line x1="${x}" y1="${KENAR.ust}" x2="${x}" y2="${YUKSEKLIK - KENAR.alt}" stroke="#d9d9d9" stroke-width="0.6"/>`;
        s += `<text x="${x}" y="${YUKSEKLIK - KENAR.alt + 17}" text-anchor="middle" class="tik">${sayiYaz(v)}</text>`;
    }
    // Eksen çizgileri
    s += `<line x1="${KENAR.sol}" y1="${YUKSEKLIK - KENAR.alt}" x2="${GENISLIK - KENAR.sag}" y2="${YUKSEKLIK - KENAR.alt}" stroke="#555" stroke-width="1"/>`;
    s += `<line x1="${KENAR.sol}" y1="${KENAR.ust}" x2="${KENAR.sol}" y2="${YUKSEKLIK - KENAR.alt}" stroke="#555" stroke-width="1"/>`;
    // Başlık ve eksen etiketleri
    s += `<text x="${GENISLIK / 2}" y="22" text-anchor="middle" class="baslik">${xmlKacis(baslik)}</text>`;
    s += `<text x="${GENISLIK / 2}" y="${YUKSEKLIK - 12}" text-anchor="middle" class="etiket">${xmlKacis(xEksen.etiket)}</text>`;
    s += `<text x="16" y="${KENAR.ust + (YUKSEKLIK - KENAR.ust - KENAR.alt) / 2}" text-anchor="middle" class="etiket" transform="rotate(-90 16 ${KENAR.ust + (YUKSEKLIK - KENAR.ust - KENAR.alt) / 2})">${xmlKacis(yEksen.etiket)}</text>`;

    return { s, xOlcek, yOlcek, icW, icH };
}

function svgSar(icerik) {
    return `<svg viewBox="0 0 ${GENISLIK} ${YUKSEKLIK}" xmlns="http://www.w3.org/2000/svg" role="img">${icerik}</svg>`;
}

/**
 * Histogram çizer (matplotlib ax.hist karşılığı).
 * Ortalama değeri kesikli kırmızı çizgi ile işaretler (Python'daki axvline).
 */
export function histogramSvg({ degerler, bolme = 30, renk = "forestgreen", baslik, xEtiket, yEtiket, ortalamaBirim = "" }) {
    if (!degerler.length) return svgSar(`<text x="${GENISLIK / 2}" y="${YUKSEKLIK / 2}" text-anchor="middle" class="etiket">Veri yok</text>`);
    const min = Math.min(...degerler), max = Math.max(...degerler);
    const aralik = max - min || 1;
    const sayimlar = new Array(bolme).fill(0);
    for (const v of degerler) {
        let k = Math.floor(((v - min) / aralik) * bolme);
        if (k >= bolme) k = bolme - 1;
        sayimlar[k]++;
    }
    const xE = { ...eksenAdimlari(min, max), etiket: xEtiket };
    const yE = { ...eksenAdimlari(0, Math.max(...sayimlar), 5, true), etiket: yEtiket };
    yE.min = 0;
    const p = panelBaslat(baslik, xE, yE);
    let s = p.s;
    for (let k = 0; k < bolme; k++) {
        if (!sayimlar[k]) continue;
        const x1 = p.xOlcek(min + (k / bolme) * aralik);
        const x2 = p.xOlcek(min + ((k + 1) / bolme) * aralik);
        const y = p.yOlcek(sayimlar[k]);
        s += `<rect x="${x1 + 0.5}" y="${y}" width="${Math.max(x2 - x1 - 1, 0.5)}" height="${YUKSEKLIK - KENAR.alt - y}" fill="${renk}" fill-opacity="0.7" stroke="${renk}" stroke-width="0.5"/>`;
    }
    // Ortalama çizgisi + etiketi (axvline + legend karşılığı)
    const ort = degerler.reduce((a, b) => a + b, 0) / degerler.length;
    const ox = p.xOlcek(ort);
    s += `<line x1="${ox}" y1="${KENAR.ust}" x2="${ox}" y2="${YUKSEKLIK - KENAR.alt}" stroke="#cc2b2b" stroke-width="1.4" stroke-dasharray="5,4"/>`;
    s += `<text x="${GENISLIK - KENAR.sag - 6}" y="${KENAR.ust + 16}" text-anchor="end" class="lejant" fill="#cc2b2b">Ortalama: ${ort.toFixed(1)}${xmlKacis(ortalamaBirim)}</text>`;
    return svgSar(s);
}

/**
 * Saçılım grafiği çizer (matplotlib ax.scatter karşılığı).
 * 1:1 doğrusu ve istatistik kutusu ile — doğruluk değerlendirmesi için.
 */
export function sacilimSvg({ x, y, baslik, xEtiket, yEtiket, birEBir = true, kutu = "" }) {
    const enBuyuk = Math.max(...x, ...y) * 1.05 || 1;
    const xE = { ...eksenAdimlari(0, enBuyuk), etiket: xEtiket };
    const yE = { ...eksenAdimlari(0, enBuyuk), etiket: yEtiket };
    xE.min = 0; yE.min = 0; xE.max = yE.max = Math.max(xE.max, yE.max);
    const p = panelBaslat(baslik, xE, yE);
    let s = p.s;
    if (birEBir) {
        // 1:1 doğrusu (mükemmel uyum çizgisi)
        s += `<line x1="${p.xOlcek(0)}" y1="${p.yOlcek(0)}" x2="${p.xOlcek(xE.max)}" y2="${p.yOlcek(yE.max)}" stroke="#8c8c8c" stroke-width="1.4" stroke-dasharray="6,4"/>`;
        s += `<text x="${p.xOlcek(xE.max * 0.86)}" y="${p.yOlcek(yE.max * 0.90)}" class="lejant" fill="#666" text-anchor="middle" transform="rotate(-45 ${p.xOlcek(xE.max * 0.86)} ${p.yOlcek(yE.max * 0.90)})">1:1 doğrusu</text>`;
    }
    for (let i = 0; i < x.length; i++) {
        s += `<circle cx="${p.xOlcek(x[i])}" cy="${p.yOlcek(y[i])}" r="4.5" fill="forestgreen" stroke="white" stroke-width="0.8"/>`;
    }
    if (kutu) {
        const satirlar = kutu.split("\n");
        const kx = KENAR.sol + 10, ky = KENAR.ust + 10;
        s += `<rect x="${kx - 6}" y="${ky - 4}" width="128" height="${satirlar.length * 17 + 10}" rx="6" fill="white" fill-opacity="0.92" stroke="#cccccc"/>`;
        satirlar.forEach((satir, i) => {
            s += `<text x="${kx}" y="${ky + 13 + i * 17}" class="lejant" fill="#333">${xmlKacis(satir)}</text>`;
        });
    }
    return svgSar(s);
}

/**
 * Çizgi grafik çizer (matplotlib ax.plot karşılığı).
 * Her noktanın üzerine değer etiketi yazar.
 */
export function cizgiSvg({ x, y, baslik, xEtiket, yEtiket, renk = "forestgreen", degerBicim = (v) => String(v) }) {
    const xE = { min: Math.min(...x), max: Math.max(...x), cizgiler: x.slice(), etiket: xEtiket };
    const yAralik = eksenAdimlari(Math.min(...y), Math.max(...y));
    const yE = { ...yAralik, etiket: yEtiket };
    // Değer etiketleri için üstte pay bırak
    yE.max += (yE.max - yE.min) * 0.12 || 1;
    const p = panelBaslat(baslik, xE, yE);
    let s = p.s;
    const nokta = x.map((xv, i) => `${p.xOlcek(xv)},${p.yOlcek(y[i])}`).join(" ");
    s += `<polyline points="${nokta}" fill="none" stroke="${renk}" stroke-width="2"/>`;
    for (let i = 0; i < x.length; i++) {
        s += `<circle cx="${p.xOlcek(x[i])}" cy="${p.yOlcek(y[i])}" r="5" fill="${renk}" stroke="white" stroke-width="1"/>`;
        s += `<text x="${p.xOlcek(x[i])}" y="${p.yOlcek(y[i]) - 11}" text-anchor="middle" class="lejant" fill="#333">${xmlKacis(degerBicim(y[i]))}</text>`;
    }
    return svgSar(s);
}

/**
 * SVG panellerini kendi kendine yeten bir HTML dosyasına kaydeder
 * (matplotlib plt.savefig karşılığı).
 *
 * @param {string} dosyaYolu - Kaydedilecek .html dosyası
 * @param {string} sayfaBasligi - Sayfanın üst başlığı
 * @param {string[]} svgler - Panel SVG'leri (2 sütunlu ızgarada dizilir)
 */
export function grafikSayfasiKaydet(dosyaYolu, sayfaBasligi, svgler) {
    const html = `<!doctype html>
<html lang="tr">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${xmlKacis(sayfaBasligi)}</title>
<style>
  body{font-family:-apple-system,"Segoe UI",Roboto,Arial,sans-serif;background:#fafaf8;color:#222;margin:0;padding:28px}
  h1{font-size:19px;text-align:center;margin:0 0 20px}
  .izgara{display:grid;grid-template-columns:repeat(auto-fit,minmax(400px,1fr));gap:18px;max-width:1080px;margin:0 auto}
  .panel{background:#fff;border:1px solid #e2e2dc;border-radius:10px;padding:10px}
  svg{width:100%;height:auto;display:block}
  .baslik{font-size:14px;font-weight:700;fill:#222}
  .etiket{font-size:12px;fill:#444}
  .tik{font-size:10.5px;fill:#666}
  .lejant{font-size:11px}
  footer{max-width:1080px;margin:16px auto 0;color:#888;font-size:12px;text-align:center}
</style>
</head>
<body>
<h1>${xmlKacis(sayfaBasligi)}</h1>
<div class="izgara">
${svgler.map((s) => `<div class="panel">${s}</div>`).join("\n")}
</div>
<footer>YTÜ Harita Mühendisliği Yüksek Lisans Tezi — grafik dosyası otomatik üretilmiştir.
Tez için PNG gerekirse tarayıcıda açıp ekran görüntüsü alın.</footer>
</body>
</html>
`;
    fs.writeFileSync(dosyaYolu, html, "utf-8");
}
