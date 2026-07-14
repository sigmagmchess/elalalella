/*
  ÇAM-AI çekirdek testleri (mini_ai.html içindeki DOM'suz katman)
  Çalıştırma:  node test/cekirdek_test.mjs
  Kapsam: TIFF çözücü (şerit/karo, LZW+öngörücü, Deflate, PackBits, 16-bit,
  float+nodata, BigTIFF, GeoTIFF etiketleri, çok sayfa, küçültme) ve ML motoru
  (LZW gidiş-dönüş, CV/Oto-AI, serileştirme, harita sınıflandırma).
*/
import { readFileSync } from 'node:fs';
import { deflateSync } from 'node:zlib';
import { fileURLToPath } from 'node:url';
import { dirname, join } from 'node:path';

const kok = dirname(dirname(fileURLToPath(import.meta.url)));
const html = readFileSync(join(kok, 'mini_ai.html'), 'utf8');
const es = html.match(/\/\*__CEKIRDEK_BASLA__\*\/([\s\S]*?)\/\*__CEKIRDEK_BITIR__\*\//);
if (!es) { console.error('Çekirdek işaretleyicileri bulunamadı!'); process.exit(1); }

const cekirdek = new Function(es[1] + `;
  return { mulberry32, karistir, lzwAc, packbitsAc, deflateAc, tiffCoz,
    ozellikCikarici, OZELLIK_SAYISI, OZELLIK_KUMELERI, OZELLIK_ADLARI,
    standartlastiriciKur, kolonSec, knnKur, softmaksEgit, ysaEgit, ormanEgit,
    karisiklikOlc, dogrulukVeF1, katmanliKatlar, fisherSkorlari,
    otoAyarListesi, otoEgit, modelPaketle, modelAc,
    haritaSiniflandir, probYumusat, bagliBilesenSay, agacIsaretle, yogunlukHaritasi,
    cokgenDoldur, tasmaDoldur };`)();
const C = cekirdek;

let gecti = 0, kaldi = 0;
function dogrula(kosul, ad){
  if (kosul){ gecti++; console.log('  ✓ ' + ad); }
  else { kaldi++; console.error('  ✗ BAŞARISIZ: ' + ad); }
}
function bolum(ad){ console.log('\n— ' + ad); }

/* ================= TIFF üretici yardımcıları ================= */
const TIP_BOY = { 2:1, 3:2, 4:4, 11:4, 12:8, 16:8 };

function lzwSifrele(veri){
  // TIFF-uyumlu LZW kodlayıcı (MSB-önce, erken geçiş; libtiff eşleniği)
  const cikti = [];
  let tampon = 0, bitler = 0;
  let kodBoyu = 9, sonraki = 258;
  const tablo = new Map();
  function yazKod(k){
    tampon = (tampon << kodBoyu) | k; bitler += kodBoyu;
    while (bitler >= 8){ bitler -= 8; cikti.push((tampon >>> bitler) & 255); }
  }
  function sifirla(){ tablo.clear(); kodBoyu = 9; sonraki = 258; }
  yazKod(256); sifirla();
  let omega = -1;
  for (let i = 0; i < veri.length; i++){
    const K = veri[i];
    if (omega < 0){ omega = K; continue; }
    const anahtar = omega * 256 + K;
    const kod = tablo.get(anahtar);
    if (kod !== undefined){ omega = kod; continue; }
    yazKod(omega);
    tablo.set(anahtar, sonraki++);
    if (sonraki === (1 << kodBoyu)) kodBoyu++;        // kodlayıcı geç geçiş (çözücü erken)
    omega = K;
    if (sonraki >= 4093){ yazKod(omega); yazKod(256); sifirla(); omega = -1; }
  }
  if (omega >= 0) yazKod(omega);
  yazKod(257);
  if (bitler > 0){ tampon <<= (8 - bitler); cikti.push(tampon & 255); }
  return new Uint8Array(cikti);
}

function packbitsSifrele(veri){
  const cikti = [];
  let i = 0;
  while (i < veri.length){
    let n = 1;
    while (i + n < veri.length && veri[i + n] === veri[i] && n < 127) n++;
    if (n >= 2){ cikti.push(257 - n, veri[i]); i += n; }
    else {
      let m = 1;
      while (i + m < veri.length && (i + m + 1 >= veri.length || veri[i + m] !== veri[i + m + 1]) && m < 128) m++;
      cikti.push(m - 1);
      for (let j = 0; j < m; j++) cikti.push(veri[i + j]);
      i += m;
    }
  }
  return new Uint8Array(cikti);
}

/* tek/çok sayfalı TIFF dosyası üretir
   sayfa: { girdiler:[[etiket, tip, sayilar|Uint8Array|string], ...] }
   büyük değerler ve veri blokları başlıktan sonra, IFD'ler en sonda yazılır */
function tiffOlustur({ le = true, big = false, sayfalar }){
  const parcalar = [];
  let poz = big ? 16 : 8;
  function baytEkle(u8){
    const o = poz;
    parcalar.push(u8);
    poz += u8.length;
    if (poz & 1){ parcalar.push(new Uint8Array(1)); poz++; }
    return o;
  }
  function sayilariYaz(tip, degerler){
    const b = Buffer.alloc(TIP_BOY[tip] * degerler.length);
    degerler.forEach((v, i) => {
      const o = i * TIP_BOY[tip];
      if (tip === 3) le ? b.writeUInt16LE(v, o) : b.writeUInt16BE(v, o);
      else if (tip === 4) le ? b.writeUInt32LE(v, o) : b.writeUInt32BE(v, o);
      else if (tip === 11) le ? b.writeFloatLE(v, o) : b.writeFloatBE(v, o);
      else if (tip === 12) le ? b.writeDoubleLE(v, o) : b.writeDoubleBE(v, o);
      else if (tip === 16) le ? b.writeBigUInt64LE(BigInt(v), o) : b.writeBigUInt64BE(BigInt(v), o);
    });
    return new Uint8Array(b);
  }
  const sayfaIfdleri = [];
  for (const sayfa of sayfalar){
    const girdiler = [];
    for (const [etiket, tip, deger] of sayfa.girdiler){
      let bayt, sayi;
      if (typeof deger === 'string'){
        const s = Buffer.from(deger + '\0', 'latin1');
        bayt = new Uint8Array(s); sayi = s.length;
      } else if (deger instanceof Uint8Array){ bayt = deger; sayi = deger.length; }
      else { bayt = sayilariYaz(tip, deger); sayi = deger.length; }
      const sinir = big ? 8 : 4;
      let inl = null, ofs = 0;
      if (bayt.length <= sinir){ inl = bayt; } else { ofs = baytEkle(bayt); }
      girdiler.push({ etiket, tip, sayi, inl, ofs });
    }
    girdiler.sort((a, b) => a.etiket - b.etiket);
    sayfaIfdleri.push(girdiler);
  }
  const ifdKonumlari = [];
  let ifdPoz = poz;
  for (const girdiler of sayfaIfdleri){
    ifdKonumlari.push(ifdPoz);
    ifdPoz += big ? (8 + girdiler.length * 20 + 8) : (2 + girdiler.length * 12 + 4);
  }
  const ifdBaytlari = [];
  sayfaIfdleri.forEach((girdiler, si) => {
    const boy = big ? (8 + girdiler.length * 20 + 8) : (2 + girdiler.length * 12 + 4);
    const b = Buffer.alloc(boy);
    let p = 0;
    if (big){ le ? b.writeBigUInt64LE(BigInt(girdiler.length), 0) : b.writeBigUInt64BE(BigInt(girdiler.length), 0); p = 8; }
    else { le ? b.writeUInt16LE(girdiler.length, 0) : b.writeUInt16BE(girdiler.length, 0); p = 2; }
    for (const g of girdiler){
      le ? b.writeUInt16LE(g.etiket, p) : b.writeUInt16BE(g.etiket, p);
      le ? b.writeUInt16LE(g.tip, p + 2) : b.writeUInt16BE(g.tip, p + 2);
      if (big) le ? b.writeBigUInt64LE(BigInt(g.sayi), p + 4) : b.writeBigUInt64BE(BigInt(g.sayi), p + 4);
      else le ? b.writeUInt32LE(g.sayi, p + 4) : b.writeUInt32BE(g.sayi, p + 4);
      const dOfs = p + (big ? 12 : 8);
      if (g.inl) Buffer.from(g.inl).copy(b, dOfs);
      else if (big) le ? b.writeBigUInt64LE(BigInt(g.ofs), dOfs) : b.writeBigUInt64BE(BigInt(g.ofs), dOfs);
      else le ? b.writeUInt32LE(g.ofs, dOfs) : b.writeUInt32BE(g.ofs, dOfs);
      p += big ? 20 : 12;
    }
    const sonraki = si + 1 < ifdKonumlari.length ? ifdKonumlari[si + 1] : 0;
    if (big) le ? b.writeBigUInt64LE(BigInt(sonraki), p) : b.writeBigUInt64BE(BigInt(sonraki), p);
    else le ? b.writeUInt32LE(sonraki, p) : b.writeUInt32BE(sonraki, p);
    ifdBaytlari.push(new Uint8Array(b));
  });
  const baslik = Buffer.alloc(big ? 16 : 8);
  le ? baslik.writeUInt16LE(0x4949, 0) : baslik.writeUInt16BE(0x4D4D, 0);
  if (big){
    le ? baslik.writeUInt16LE(43, 2) : baslik.writeUInt16BE(43, 2);
    le ? baslik.writeUInt16LE(8, 4) : baslik.writeUInt16BE(8, 4);
    le ? baslik.writeBigUInt64LE(BigInt(ifdKonumlari[0]), 8) : baslik.writeBigUInt64BE(BigInt(ifdKonumlari[0]), 8);
  } else {
    le ? baslik.writeUInt16LE(42, 2) : baslik.writeUInt16BE(42, 2);
    le ? baslik.writeUInt32LE(ifdKonumlari[0], 4) : baslik.writeUInt32BE(ifdKonumlari[0], 4);
  }
  const hepsi = [new Uint8Array(baslik), ...parcalar, ...ifdBaytlari];
  const toplam = hepsi.reduce((a, u) => a + u.length, 0);
  const cikti = new Uint8Array(toplam);
  let o = 0;
  for (const u of hepsi){ cikti.set(u, o); o += u.length; }
  return cikti.buffer.slice(cikti.byteOffset, cikti.byteOffset + cikti.byteLength);
  // Not: cikti kendi buffer'ına sahip (offset 0) — dilim güvenli
}

/* deterministik test deseni */
function desenRGB(w, h, bant){
  const veri = new Uint8Array(w * h * bant);
  for (let y = 0; y < h; y++)
    for (let x = 0; x < w; x++){
      const p = (y * w + x) * bant;
      veri[p] = (x * 7 + y * 3) & 255;
      if (bant > 1) veri[p + 1] = (x * 2 + y * 11) & 255;
      if (bant > 2) veri[p + 2] = (x ^ y) & 255;
      if (bant > 3) veri[p + 3] = x < w / 2 ? 255 : 0;   // sağ yarı saydam
    }
  return veri;
}
function seritler(veri, w, h, bant, satirBasina){
  const dilimler = [];
  for (let y0 = 0; y0 < h; y0 += satirBasina){
    const y1 = Math.min(h, y0 + satirBasina);
    dilimler.push(veri.subarray(y0 * w * bant, y1 * w * bant));
  }
  return dilimler;
}
function ongoruUygula(veri, w, h, bant){
  const kopya = veri.slice();
  for (let y = 0; y < h; y++){
    const b0 = y * w * bant;
    for (let i = w * bant - 1; i >= bant; i--)
      kopya[b0 + i] = (kopya[b0 + i] - kopya[b0 + i - bant]) & 255;
  }
  return kopya;
}
function rgbaKarsilastir(sonuc, veri, w, h, bant){
  let fark = 0;
  for (let i = 0; i < w * h; i++){
    const p = i * 4, q = i * bant;
    if (sonuc.rgba[p] !== veri[q]) fark++;
    if (sonuc.rgba[p + 1] !== veri[bant > 1 ? q + 1 : q]) fark++;
    if (sonuc.rgba[p + 2] !== veri[bant > 2 ? q + 2 : q]) fark++;
  }
  return fark;
}
function tabanGirdiler(w, h, bant, bit, sik, ekstra){
  const g = [
    [256, 3, [w]], [257, 3, [h]],
    [258, 3, new Array(bant).fill(bit)],
    [259, 3, [sik]], [262, 3, [bant >= 3 ? 2 : 1]],
    [277, 3, [bant]], [284, 3, [1]]
  ];
  return g.concat(ekstra || []);
}

/* ================= 1. LZW gidiş-dönüş ================= */
bolum('LZW kodlayıcı/çözücü gidiş-dönüş');
{
  const rng = C.mulberry32(7);
  const durumlar = [
    ['rastgele 5000 bayt (9→10→11 bit geçişleri)', Uint8Array.from({ length: 5000 }, () => (rng() * 256) | 0)],
    ['tek değer 4000 bayt (KwKwK zinciri)', new Uint8Array(4000).fill(129)],
    ['ABAB deseni', Uint8Array.from({ length: 1000 }, (_, i) => i % 2 ? 65 : 66)],
    ['kısa (3 bayt)', Uint8Array.from([1, 2, 3])],
    ['tablo dolduran 60000 bayt', Uint8Array.from({ length: 60000 }, () => (rng() * 250) | 0)]
  ];
  for (const [ad, veri] of durumlar){
    const acilan = C.lzwAc(lzwSifrele(veri), veri.length);
    let ayni = acilan.length >= veri.length;
    if (ayni) for (let i = 0; i < veri.length; i++) if (acilan[i] !== veri[i]){ ayni = false; break; }
    dogrula(ayni, 'LZW: ' + ad);
  }
  const pb = Uint8Array.from({ length: 700 }, (_, i) => i % 90 < 45 ? 7 : (i & 255));
  const pbAc = C.packbitsAc(packbitsSifrele(pb), pb.length);
  dogrula(pb.every((v, i) => pbAc[i] === v), 'PackBits gidiş-dönüş');
}

/* ================= 2. TIFF çözücü ================= */
bolum('TIFF çözücü — biçim çeşitleri');
const W = 41, H = 23;                                   // bilerek tek sayı boyutlar
{
  // T1: 8-bit RGB, sıkıştırmasız, şeritli, küçük endian
  const veri = desenRGB(W, H, 3);
  const dilim = seritler(veri, W, H, 3, 7);
  let poz = [], boy = [];
  const buf = tiffOlustur({ le: true, sayfalar: [{ girdiler: tabanGirdiler(W, H, 3, 8, 1, [
    [273, 4, dilim.map(() => 0)], [278, 3, [7]], [279, 4, dilim.map(d => d.length)]
  ]) }] });
  // ofsetleri elle yerleştirmek yerine üreticide veri bloklarını girdi olarak veriyoruz:
}
function tamTiff({ le = true, big = false, w, h, bant, bit = 8, sik = 1, satirBasina = 8,
                   ongorucu = 0, karo = null, veriUret, ekstraGirdi = [], sayfaOnu = [] }){
  const veri = veriUret ? veriUret() : desenRGB(w, h, bant);
  let parcalar = [];
  if (karo){
    const [tw, th] = karo;
    for (let ty = 0; ty < Math.ceil(h / th); ty++)
      for (let tx = 0; tx < Math.ceil(w / tw); tx++){
        const karoVeri = new Uint8Array(tw * th * bant * (bit >> 3));
        const kaynak = new Uint8Array(veri.buffer, veri.byteOffset, veri.byteLength);
        for (let y = 0; y < th; y++){
          const gy = ty * th + y;
          if (gy >= h) break;
          for (let x = 0; x < tw; x++){
            const gx = tx * tw + x;
            if (gx >= w) break;
            for (let c = 0; c < bant * (bit >> 3); c++)
              karoVeri[(y * tw + x) * bant * (bit >> 3) + c] = kaynak[(gy * w + gx) * bant * (bit >> 3) + c];
          }
        }
        parcalar.push(karoVeri);
      }
  } else {
    const kaynak = new Uint8Array(veri.buffer, veri.byteOffset, veri.byteLength);
    for (let y0 = 0; y0 < h; y0 += satirBasina){
      const y1 = Math.min(h, y0 + satirBasina);
      parcalar.push(kaynak.slice(y0 * w * bant * (bit >> 3), y1 * w * bant * (bit >> 3)));
    }
  }
  if (ongorucu === 2) parcalar = parcalar.map((p, i) => {
    const satirlar = karo ? karo[1] : Math.min(satirBasina, h - i * satirBasina);
    return ongoruUygula(p, karo ? karo[0] : w, satirlar, bant);
  });
  let sikili = parcalar;
  if (sik === 5) sikili = parcalar.map(p => lzwSifrele(p));
  else if (sik === 8) sikili = parcalar.map(p => new Uint8Array(deflateSync(p)));
  else if (sik === 32773) sikili = parcalar.map(p => packbitsSifrele(p));
  const konumGirdi = karo
    ? [[322, 3, [karo[0]]], [323, 3, [karo[1]]], [324, 4, sikili.map(() => 0)], [325, 4, sikili.map(s => s.length)]]
    : [[273, 4, sikili.map(() => 0)], [278, 3, [satirBasina]], [279, 4, sikili.map(s => s.length)]];
  const girdiler = tabanGirdiler(w, h, bant, bit, sik,
    konumGirdi.concat(ongorucu ? [[317, 3, [ongorucu]]] : []).concat(ekstraGirdi));
  // iki geçiş: önce ofsetleri 0 ile kur, üretilen dosyada blokları sona ekleyip ofsetleri yamala
  const govde = tiffOlustur({ le, big, sayfalar: sayfaOnu.concat([{ girdiler }]) });
  const gBuf = Buffer.from(govde.slice(0));
  let toplamEk = 0;
  const ofsetler = sikili.map(s => { const o = gBuf.length + toplamEk + (toplamEk & 1); toplamEk += s.length + (s.length & 1); return o; });
  const ekler = [];
  sikili.forEach(s => { ekler.push(Buffer.from(s)); if (s.length & 1) ekler.push(Buffer.alloc(1)); });
  const tam = Buffer.concat([gBuf, ...ekler]);
  // StripOffsets/TileOffsets girdisini bul ve gerçek ofsetlerle doldur
  const hedefEtiket = karo ? 324 : 273;
  yamala(tam, le, big, hedefEtiket, ofsetler);
  return { buf: tam.buffer.slice(tam.byteOffset, tam.byteOffset + tam.byteLength), veri };
}
function yamala(buf, le, big, etiket, degerler){
  // IFD'leri gezip verilen LONG dizili etiketi bul, değer alanını güncelle
  let ifd = big
    ? Number(le ? buf.readBigUInt64LE(8) : buf.readBigUInt64BE(8))
    : (le ? buf.readUInt32LE(4) : buf.readUInt32BE(4));
  while (ifd){
    const adet = big ? Number(le ? buf.readBigUInt64LE(ifd) : buf.readBigUInt64BE(ifd))
                     : (le ? buf.readUInt16LE(ifd) : buf.readUInt16BE(ifd));
    let p = ifd + (big ? 8 : 2);
    for (let i = 0; i < adet; i++, p += big ? 20 : 12){
      const t = le ? buf.readUInt16LE(p) : buf.readUInt16BE(p);
      if (t !== etiket) continue;
      const sayi = big ? Number(le ? buf.readBigUInt64LE(p + 4) : buf.readBigUInt64BE(p + 4))
                       : (le ? buf.readUInt32LE(p + 4) : buf.readUInt32BE(p + 4));
      const dOfs = p + (big ? 12 : 8);
      const sinir = big ? 8 : 4;
      let hedef;
      if (sayi * 4 <= sinir) hedef = dOfs;
      else hedef = big ? Number(le ? buf.readBigUInt64LE(dOfs) : buf.readBigUInt64BE(dOfs))
                       : (le ? buf.readUInt32LE(dOfs) : buf.readUInt32BE(dOfs));
      degerler.forEach((v, j) => {
        le ? buf.writeUInt32LE(v, hedef + j * 4) : buf.writeUInt32BE(v, hedef + j * 4);
      });
      return;
    }
    ifd = big ? Number(le ? buf.readBigUInt64LE(p) : buf.readBigUInt64BE(p))
              : (le ? buf.readUInt32LE(p) : buf.readUInt32BE(p));
  }
  throw new Error('yamalanacak etiket bulunamadı: ' + etiket);
}

{
  const t1 = tamTiff({ w: W, h: H, bant: 3 });
  const s1 = await C.tiffCoz(t1.buf, { maxDim: 4096 });
  dogrula(s1.w === W && s1.h === H && s1.outW === W && s1.outH === H, 'T1 sıkıştırmasız şerit: boyutlar');
  dogrula(rgbaKarsilastir(s1, t1.veri, W, H, 3) === 0, 'T1 sıkıştırmasız şerit: piksel eşleşmesi');
  dogrula(s1.gecerli.every(v => v === 1), 'T1: tüm pikseller geçerli');

  const t2 = tamTiff({ le: false, w: W, h: H, bant: 3, sik: 5, ongorucu: 2, satirBasina: 5 });
  const s2 = await C.tiffCoz(t2.buf, { maxDim: 4096 });
  dogrula(rgbaKarsilastir(s2, t2.veri, W, H, 3) === 0, 'T2 büyük-endian LZW + öngörücü-2 şerit');

  const t3 = tamTiff({ w: W, h: H, bant: 4, sik: 5, karo: [16, 16] });
  const s3 = await C.tiffCoz(t3.buf, { maxDim: 4096 });
  dogrula(rgbaKarsilastir(s3, t3.veri, W, H, 4) === 0, 'T3 karolu LZW RGBA: piksel eşleşmesi');
  {
    let solGecerli = true, sagGecersiz = true;
    for (let y = 0; y < H; y++){
      if (!s3.gecerli[y * W + 2]) solGecerli = false;
      if (s3.gecerli[y * W + W - 2]) sagGecersiz = false;
    }
    dogrula(solGecerli && sagGecersiz, 'T3: alfa=0 bölgesi "veri yok" sayıldı');
  }

  const t4 = tamTiff({ w: W, h: H, bant: 3, sik: 8 });
  const s4 = await C.tiffCoz(t4.buf, { maxDim: 4096 });
  dogrula(rgbaKarsilastir(s4, t4.veri, W, H, 3) === 0, 'T4 Deflate/ZIP şerit');

  const t5 = tamTiff({ w: W, h: H, bant: 1, sik: 32773 });
  const s5 = await C.tiffCoz(t5.buf, { maxDim: 4096 });
  dogrula(rgbaKarsilastir(s5, t5.veri, W, H, 1) === 0, 'T5 PackBits gri');

  const t6 = tamTiff({ big: true, w: W, h: H, bant: 3, sik: 5, ongorucu: 2 });
  const s6 = await C.tiffCoz(t6.buf, { maxDim: 4096 });
  dogrula(rgbaKarsilastir(s6, t6.veri, W, H, 3) === 0, 'T6 BigTIFF LZW + öngörücü');

  // T7: 16-bit gri gradyan (deflate) → gerdirme sonrası tekdüze artan olmalı
  const t7 = tamTiff({ w: 64, h: 8, bant: 1, bit: 16, sik: 8, satirBasina: 4, veriUret(){
    const v = new Uint16Array(64 * 8);
    for (let y = 0; y < 8; y++) for (let x = 0; x < 64; x++) v[y * 64 + x] = 1000 + x * 500;
    return v;
  } });
  const s7 = await C.tiffCoz(t7.buf, { maxDim: 4096 });
  let artan = true;
  for (let x = 2; x < 62; x++)
    if (s7.rgba[(3 * 64 + x) * 4] < s7.rgba[(3 * 64 + x - 1) * 4] - 1) artan = false;
  dogrula(artan && s7.rgba[(3 * 64 + 40) * 4] > 60, 'T7 16-bit gri: yüzdelik gerdirme tekdüze');

  // T8: float32 DSM + GDAL_NODATA
  const t8 = tamTiff({ w: 32, h: 16, bant: 1, bit: 32, satirBasina: 16,
    ekstraGirdi: [[339, 3, [3]], [42113, 2, '-10000']],
    veriUret(){
      const v = new Float32Array(32 * 16);
      for (let i = 0; i < v.length; i++) v[i] = 50 + (i % 32);
      for (let y = 0; y < 16; y++) for (let x = 0; x < 6; x++) v[y * 32 + x] = -10000;
      return v;
    } });
  const s8 = await C.tiffCoz(t8.buf, { maxDim: 4096 });
  let nodataDogru = true, veriDogru = true;
  for (let y = 0; y < 16; y++){
    for (let x = 0; x < 6; x++) if (s8.gecerli[y * 32 + x]) nodataDogru = false;
    for (let x = 8; x < 32; x++) if (!s8.gecerli[y * 32 + x]) veriDogru = false;
  }
  dogrula(nodataDogru && veriDogru, 'T8 float32 DSM: GDAL_NODATA maskesi');

  // T9: küçültme (k=4, 2×2 ortalama) — sabit renkli görüntü aynen kalmalı
  const t9 = tamTiff({ w: 200, h: 100, bant: 3, satirBasina: 16, veriUret(){
    const v = new Uint8Array(200 * 100 * 3);
    for (let i = 0; i < 200 * 100; i++){ v[i * 3] = 30; v[i * 3 + 1] = 160; v[i * 3 + 2] = 70; }
    return v;
  } });
  const s9 = await C.tiffCoz(t9.buf, { maxDim: 50 });
  dogrula(s9.olcek === 4 && s9.outW === 50 && s9.outH === 25, 'T9 küçültme: 1/4 ölçek boyutları');
  let renkTamam = true;
  for (let i = 0; i < s9.outW * s9.outH; i++){
    const p = i * 4;
    if (s9.rgba[p] !== 30 || s9.rgba[p + 1] !== 160 || s9.rgba[p + 2] !== 70) renkTamam = false;
  }
  dogrula(renkTamam && s9.gecerli.every(v => v === 1), 'T9 küçültme: renkler korunur');

  // T10: GeoTIFF etiketleri (EPSG:32635, GSD 5 cm)
  const t10 = tamTiff({ w: W, h: H, bant: 3, ekstraGirdi: [
    [33550, 12, [0.05, 0.05, 0]],
    [33922, 12, [0, 0, 0, 411000.25, 4545000.75, 0]],
    [34735, 3, [1, 1, 0, 3, 1024, 0, 1, 1, 3072, 0, 1, 32635, 1026, 34737, 8, 0]],
    [34737, 2, 'WGS8435N']
  ] });
  const s10 = await C.tiffCoz(t10.buf, { maxDim: 4096 });
  dogrula(!!s10.geo && Math.abs(s10.geo.sx - 0.05) < 1e-12 && s10.geo.epsg === 32635 &&
    Math.abs(s10.geo.ox - 411000.25) < 1e-9 && Math.abs(s10.geo.oy - 4545000.75) < 1e-9 &&
    !s10.geo.cografi && s10.geo.ad === 'WGS8435N', 'T10 GeoTIFF: GSD/EPSG/köşe koordinatı');

  // T11: çok sayfa — önce küçültülmüş kopya, sonra tam çözünürlük → tam olan seçilmeli
  const kucukVeri = desenRGB(10, 6, 3);
  const kucukSayfa = { girdiler: tabanGirdiler(10, 6, 3, 8, 1, [
    [254, 4, [1]], [273, 4, [0]], [278, 3, [6]], [279, 4, [kucukVeri.length]]
  ]) };
  const t11 = tamTiff({ w: W, h: H, bant: 3, sayfaOnu: [kucukSayfa] });
  const s11 = await C.tiffCoz(t11.buf, { maxDim: 4096 });
  dogrula(s11.w === W && s11.h === H, 'T11 çok sayfa: küçültülmüş kopya atlandı');

  // T12: bozuk dosyalar anlaşılır hata vermeli
  let hataMesaji = '';
  try{ await C.tiffCoz(new Uint8Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 0, 0, 0, 0, 0, 0, 0]).buffer, {}); }
  catch(h){ hataMesaji = h.message; }
  dogrula(/TIFF imzası/.test(hataMesaji), 'T12 bozuk dosya: anlaşılır Türkçe hata');
}

/* ================= 3. Öznitelikler ================= */
bolum('Öznitelik çıkarımı');
{
  const w = 64, h = 64;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const gecerli = new Uint8Array(w * h).fill(1);
  for (let i = 0; i < w * h; i++){
    const x = i % w;
    const p = i * 4;
    if (x < 32){ rgba[p] = 40; rgba[p + 1] = 170; rgba[p + 2] = 50; }   // bitki yeşili
    else { rgba[p] = 150; rgba[p + 1] = 150; rgba[p + 2] = 150; }        // gri zemin
    rgba[p + 3] = 255;
  }
  const cikar = C.ozellikCikarici(rgba, gecerli, w, h, 16);
  const ozY = new Float64Array(24), ozG = new Float64Array(24);
  dogrula(cikar(16, 32, ozY) && cikar(48, 32, ozG), 'iki yamada çıkarım başarılı');
  dogrula(ozY[12] > 0.15 && ozG[12] < 0.02, 'ExG: bitki > zemin');
  dogrula(ozY[13] < 0.01 && ozY[21] < 0.01, 'tekdüze yamada std ≈ 0');
  dogrula(ozY[1] > ozY[0] && ozY[1] > ozY[2], 'yeşil bant baskın');
  const ozKenar = new Float64Array(24);
  dogrula(cikar(32, 32, ozKenar) && ozKenar[20] > 0.001 && ozKenar[22] > 0.1 &&
    ozKenar[22] > ozY[22] + 0.05,
    'sınır yamasında doku (gradyan + aralık) yüksek');
  gecerli.fill(0);
  dogrula(!cikar(16, 32, ozY), 'geçersiz alanda çıkarım reddedilir');
}

/* ince ayrım kanalları: iğne benzeri (yüksek frekanslı benekli) doku ile
   geniş yaprak benzeri (yumuşak lekeli) dokuyu ayırt edebiliyor mu? */
bolum('İnce doku kanalları — iğne / geniş yaprak duyarlılığı');
{
  const w = 96, h = 48;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const gecerli = new Uint8Array(w * h).fill(1);
  for (let y = 0; y < h; y++){
    for (let x = 0; x < w; x++){
      const p = (y * w + x) * 4;
      /* iki yarının RENK ortalaması aynı (R40, G~130, B50) — yalnız DOKU farklı,
         böylece Fisher'ın keşfi renkten değil dokudan gelmek zorunda */
      if (x < 48){
        const benek = ((x * 7 + y * 13 + ((x * x + y) % 5)) % 3) === 0 ? -60 : 30;
        rgba[p] = 40; rgba[p + 1] = 130 + benek; rgba[p + 2] = 50;
      } else {
        const leke = Math.round(35 * Math.sin(x / 4) * Math.sin(y / 4));
        rgba[p] = 40; rgba[p + 1] = 130 + leke; rgba[p + 2] = 50;
      }
      rgba[p + 3] = 255;
    }
  }
  const cikar = C.ozellikCikarici(rgba, gecerli, w, h, 24);
  const igne = new Float64Array(36), genis = new Float64Array(36);
  dogrula(cikar(24, 24, igne) && cikar(72, 24, genis), '36 kanallı çıkarım başarılı');
  dogrula(igne[27] > genis[27] * 1.4,
    'ince doku enerjisi: iğne > geniş yaprak (' + igne[27].toFixed(3) + ' vs ' + genis[27].toFixed(3) + ')');
  dogrula(igne[29] > genis[29],
    'doku ölçek oranı iğnede yüksek (' + igne[29].toFixed(2) + ' vs ' + genis[29].toFixed(2) + ')');
  dogrula(igne[31] > genis[31],
    'GLCM kontrast iğnede yüksek (' + igne[31].toFixed(2) + ' vs ' + genis[31].toFixed(2) + ')');
  dogrula(genis[32] > igne[32],
    'GLCM homojenlik geniş yaprakta yüksek (' + genis[32].toFixed(3) + ' vs ' + igne[32].toFixed(3) + ')');
  dogrula(igne[34] > genis[34],
    'kenar yoğunluğu iğnede yüksek (' + igne[34].toFixed(2) + ' vs ' + genis[34].toFixed(2) + ')');
  /* keşif mekanizması: Fisher bu kanalları kendiliğinden öne çıkarmalı */
  const Xd = [], yd = [];
  const oz = new Float64Array(36);
  for (let t = 0; t < 60; t++){
    const sinif = t % 2;
    const cx = sinif === 0 ? 13 + ((t * 3) % 22) : 61 + ((t * 3) % 22);
    if (!cikar(cx, 13 + ((t * 5) % 22), oz)) continue;
    for (let j = 0; j < 36; j++) Xd.push(oz[j]);
    yd.push(sinif);
  }
  const fisher = C.fisherSkorlari(Float32Array.from(Xd), Int32Array.from(yd), yd.length, 36, 2);
  const enIyi = fisher.map((s, j) => [s, j]).sort((a, b) => b[0] - a[0]).slice(0, 6).map(p => p[1]);
  dogrula(enIyi.some(j => j >= 24),
    'Fisher, yeni doku kanallarını kendiliğinden keşfetti (ilk 6: ' + enIyi.join(',') + ')');
}

/* ================= 4. ML motoru ================= */
bolum('ML motoru — modeller ve Oto-AI');
function blobVeri(n, K, rng){
  const X = new Float32Array(n * C.OZELLIK_SAYISI);
  const y = new Int32Array(n);
  for (let i = 0; i < n; i++){
    const c = i % K;
    y[i] = c;
    for (let j = 0; j < C.OZELLIK_SAYISI; j++){
      const bilgi = j < 10 ? (c * 1.6) : 0;                 // ilk 10 boyut bilgi taşır
      const gurultu = (rng() + rng() + rng() - 1.5) * 0.5;
      X[i * C.OZELLIK_SAYISI + j] = bilgi + gurultu;
    }
  }
  return { X, y };
}
{
  const rng = C.mulberry32(11);
  const { X, y } = blobVeri(240, 2, rng);
  const katlar = C.katmanliKatlar(y, 240, 4, C.mulberry32(1));
  dogrula(katlar.length === 4 && katlar.every(k => k.length === 60), 'katmanlı katlar dengeli');
  const s0 = katlar[0].filter(i => y[i] === 0).length;
  dogrula(s0 === 30, 'katlarda sınıf oranı korunur');

  const fisher = C.fisherSkorlari(X, y, 240, C.OZELLIK_SAYISI, 2);
  const enIyi10 = fisher.map((s, j) => [s, j]).sort((a, b) => b[0] - a[0]).slice(0, 10).map(p => p[1]);
  dogrula(enIyi10.filter(j => j < 10).length >= 8, 'Fisher: bilgi taşıyan boyutlar öne çıkar');

  const OS = C.OZELLIK_SAYISI;
  const std = C.standartlastiriciKur(X, 240, OS);
  const Xs = std.donustur(X, 240);
  for (const [ad, kur] of [
    ['k-NN', () => C.knnKur(Xs, y, 240, OS, 2, 5)],
    ['Softmaks', () => C.softmaksEgit(Xs, y, 240, OS, 2, { devir: 80 }, C.mulberry32(3))],
    ['YSA', () => C.ysaEgit(Xs, y, 240, OS, 2, { gizli: [16], devir: 60 }, C.mulberry32(3))],
    ['R.Orman', () => C.ormanEgit(Xs, y, 240, OS, 2, { agac: 25, derinlik: 8 }, C.mulberry32(3))]
  ]){
    const model = await kur();
    const x = new Float64Array(OS), p = new Float32Array(2);
    let dogru = 0;
    for (let i = 0; i < 240; i++){
      for (let j = 0; j < OS; j++) x[j] = Xs[i * OS + j];
      model.tahminProba(x, p);
      if ((p[1] > p[0] ? 1 : 0) === y[i]) dogru++;
      const t = p[0] + p[1];
      if (Math.abs(t - 1) > 1e-3) { dogru = -1000; break; }
    }
    dogrula(dogru / 240 > 0.9, ad + ': eğitim doğruluğu > %90 ve olasılıklar toplamı 1');
  }
}
{
  const rng = C.mulberry32(21);
  const { X, y } = blobVeri(180, 3, rng);
  const sonuc = await C.otoEgit(X, y, 180, 3, { katSayi: 3 });
  dogrula(sonuc.liderlik.length === C.otoAyarListesi().length, 'Oto-AI: tüm kombinasyonlar denendi (' + sonuc.liderlik.length + ')');
  dogrula(C.otoAyarListesi(true).length > C.otoAyarListesi().length,
    'geniş arama daha çok kombinasyon dener (' + C.otoAyarListesi(true).length + ')');
  dogrula(sonuc.enIyi.cv.f1 > 0.85, 'Oto-AI: kazanan F1 > 0,85 (gerçek: ' + sonuc.enIyi.cv.f1.toFixed(3) + ')');
  dogrula(sonuc.liderlik.every((r, i, a) => i === 0 || a[i - 1].f1 >= r.f1), 'lider tablosu sıralı');
  const sonuc2 = await C.otoEgit(X, y, 180, 3, { katSayi: 3 });
  dogrula(sonuc.enIyi.ad === sonuc2.enIyi.ad &&
    Math.abs(sonuc.enIyi.cv.f1 - sonuc2.enIyi.cv.f1) < 1e-12, 'Oto-AI: tekrarlanabilir (tohum=42)');

  // serileştirme gidiş-dönüşü — 4 model tipinin hepsi
  const siniflar = [{ ad: 'Çam', renk: '#2fa84f' }, { ad: 'Diğer', renk: '#e2574c' }, { ad: 'Üçüncü', renk: '#3b7dd8' }];
  for (const tipAdayi of ['knn', 'softmaks', 'ysa', 'orman']){
    const aday = sonuc.liderlik.find(r => {
      const ad = r.ad.toLowerCase();
      return tipAdayi === 'knn' ? ad.includes('k-nn')
        : tipAdayi === 'softmaks' ? ad.includes('softmaks')
        : tipAdayi === 'ysa' ? ad.includes('ysa') : ad.includes('orman');
    });
    if (!aday) continue;
    // adayı yeniden eğit (kazanan yolunu izleyerek)
    const kolonlar = C.OZELLIK_KUMELERI[aday.kume];
    const Xk = C.kolonSec(X, 180, kolonlar);
    const std = C.standartlastiriciKur(Xk, 180, kolonlar.length);
    const Xs = std.donustur(Xk, 180);
    const kur = C.otoAyarListesi()[aday.indeks].kur;
    const model = await kur(Xs, y, 180, kolonlar.length, 3, C.mulberry32(42));
    const paket = C.modelPaketle({ ad: aday.ad, kume: aday.kume, kolonlar, model, std,
      cv: { dogruluk: aday.dogruluk, f1: aday.f1, katSayi: 3 } }, siniflar, 24);
    const geri = C.modelAc(JSON.parse(JSON.stringify(paket)));
    const x = new Float64Array(kolonlar.length), p1 = new Float32Array(3), p2 = new Float32Array(3);
    let ayni = true;
    for (let i = 0; i < 40; i++){
      for (let j = 0; j < kolonlar.length; j++) x[j] = Xk[i * kolonlar.length + j];
      const x1 = Float64Array.from(x), x2 = Float64Array.from(x);
      std.tek(x1, x1); model.tahminProba(x1, p1);
      geri.std.tek(x2, x2); geri.model.tahminProba(x2, p2);
      for (let c = 0; c < 3; c++) if (Math.abs(p1[c] - p2[c]) > 1e-5) ayni = false;
    }
    dogrula(ayni, 'serileştirme gidiş-dönüş: ' + tipAdayi);
  }
}

/* ================= 5. Harita sınıflandırma ================= */
bolum('Harita sınıflandırma + yumuşatma + bölge sayımı');
{
  // yarı yeşil / yarı gri sentetik "ortomozaik"
  const w = 160, h = 80;
  const rgba = new Uint8ClampedArray(w * h * 4);
  const gecerli = new Uint8Array(w * h).fill(1);
  const rng = C.mulberry32(5);
  for (let i = 0; i < w * h; i++){
    const x = i % w, p = i * 4;
    const g = () => (rng() * 24 - 12) | 0;
    if (x < 80){ rgba[p] = 45 + g(); rgba[p + 1] = 165 + g(); rgba[p + 2] = 55 + g(); }
    else { rgba[p] = 140 + g(); rgba[p + 1] = 140 + g(); rgba[p + 2] = 140 + g(); }
    rgba[p + 3] = 255;
  }
  // eğitim örnekleri: her iki yarıdan yamalar
  const cikar = C.ozellikCikarici(rgba, gecerli, w, h, 16);
  const oz = new Float64Array(C.OZELLIK_SAYISI);
  const Xd = [], yd = [];
  for (let t = 0; t < 120; t++){
    const sinif = t % 2;
    const cx = sinif === 0 ? 12 + ((rng() * 56) | 0) : 92 + ((rng() * 56) | 0);
    const cy = 10 + ((rng() * 60) | 0);
    if (!cikar(cx, cy, oz)) continue;
    for (let j = 0; j < C.OZELLIK_SAYISI; j++) Xd.push(oz[j]);
    yd.push(sinif);
  }
  const X = Float32Array.from(Xd), y = Int32Array.from(yd), n = yd.length;
  const kolonlar = C.OZELLIK_KUMELERI['Renk+İndeks'];
  const Xk = C.kolonSec(X, n, kolonlar);
  const std = C.standartlastiriciKur(Xk, n, kolonlar.length);
  const model = await C.softmaksEgit(std.donustur(Xk, n), y, n, kolonlar.length, 2, { devir: 120 }, C.mulberry32(9));
  const sonuc = await C.haritaSiniflandir({
    rgba, gecerli, W: w, H: h, yama: 16, adim: 8,
    model, std, kolonlar, K: 2
  });
  let dogru = 0, toplam = 0;
  for (let gy = 0; gy < sonuc.gh; gy++)
    for (let gx = 0; gx < sonuc.gw; gx++){
      const cx = sonuc.yari + gx * sonuc.adim;
      if (Math.abs(cx - 80) < 12) continue;                  // sınır bloklarını sayma
      const s = sonuc.sinif[gy * sonuc.gw + gx];
      if (s === 255) continue;
      toplam++;
      if (s === (cx < 80 ? 0 : 1)) dogru++;
    }
  dogrula(toplam > 50 && dogru / toplam > 0.95, 'harita: bloklar %95+ doğru (' + dogru + '/' + toplam + ')');
  const puruzsuz = C.probYumusat(sonuc.probs, sonuc.gw, sonuc.gh, 2);
  dogrula(puruzsuz.length === sonuc.probs.length, 'yumuşatma boyut korur');
  const maske = new Uint8Array(sonuc.gw * sonuc.gh);
  for (let i = 0; i < maske.length; i++) maske[i] = sonuc.sinif[i] === 0 ? 1 : 0;
  const bolgeler = C.bagliBilesenSay(maske, sonuc.gw, sonuc.gh);
  dogrula(bolgeler >= 1 && bolgeler <= 3, 'bağlı bileşen: yeşil yarı ~1 bölge (' + bolgeler + ')');
}

/* ================= 6. Ağaç işaretleme ve yoğunluk ================= */
bolum('Ağaç işaretleme (NMS) ve en yoğun bölge');
{
  const gw = 40, gh = 30, K = 2;
  const probs = new Float32Array(gw * gh * K);
  const sinif = new Uint8Array(gw * gh).fill(255);
  function tac(cx, cy, r, p){
    for (let y = Math.max(0, cy - r); y <= Math.min(gh - 1, cy + r); y++)
      for (let x = Math.max(0, cx - r); x <= Math.min(gw - 1, cx + r); x++){
        const u = Math.hypot(x - cx, y - cy);
        if (u > r) continue;
        const i = y * gw + x;
        const deger = p * (1 - u / (r + 1));
        if (deger > probs[i * K]){
          probs[i * K] = deger;
          probs[i * K + 1] = 1 - deger;
          sinif[i] = deger >= 0.5 ? 0 : 1;
        }
      }
  }
  tac(8, 8, 3, 1.0);
  tac(30, 20, 3, 0.95);
  tac(34, 22, 3, 0.9);                                  // bitişik komşu taç (~4,5 blok)
  const agaclar = C.agacIsaretle({ probs, sinif, gw, gh, K, camIdx: 0, esik: 0.5, minMesafe: 4 });
  dogrula(agaclar.length === 3, 'üç taç ayrı ayrı bulundu (' + agaclar.length + ')');
  dogrula(agaclar.some(a => Math.abs(a.x - 8) <= 1 && Math.abs(a.y - 8) <= 1) &&
          agaclar.every(a => a.p >= 0.5), 'tepe noktaları doğru konumda ve eşik üstünde');
  const kaba = C.agacIsaretle({ probs, sinif, gw, gh, K, camIdx: 0, esik: 0.5, minMesafe: 8 });
  dogrula(kaba.length === 2, 'büyük taç çapı bitişik taçları birleştirir (' + kaba.length + ')');
  const maske = new Uint8Array(gw * gh);
  for (let i = 0; i < gw * gh; i++) maske[i] = (sinif[i] === 0 && probs[i * K] >= 0.5) ? 1 : 0;
  const yog = C.yogunlukHaritasi(maske, gw, gh, 5);
  dogrula(Math.hypot(yog.enX - 32, yog.enY - 21) <= 6,
    'en yoğun bölge çift taç kümesinde (' + yog.enX + ',' + yog.enY + ')');
  dogrula(yog.enDeger > 0 && yog.enDeger <= 1, 'yoğunluk 0-1 aralığında');
}

/* ================= 7. Etiketleme araçları ================= */
bolum('Etiketleme araçları: çokgen dolgusu ve sihirli değnek');
{
  const W = 60, H = 40;
  const maske = new Uint8Array(W * H);
  const yaz = C.cokgenDoldur(maske, W, H, [10, 10, 50, 10, 30, 30], 2);   // taban 40, yükseklik 20
  dogrula(yaz > 320 && yaz < 480, 'üçgen dolgu alanı makul (' + yaz + ' ≈ 400)');
  dogrula(maske[15 * W + 30] === 2 && maske[5 * W + 30] === 0 && maske[15 * W + 5] === 0,
    'üçgenin içi dolu, dışı boş');
  dogrula(C.cokgenDoldur(maske, W, H, [1, 1, 5, 5], 3) === 0, 'iki köşeli "çokgen" reddedilir');

  /* değnek: sol yarı yeşil, sağ yarı gri */
  const rgba = new Uint8ClampedArray(W * H * 4);
  const gecerli = new Uint8Array(W * H).fill(1);
  for (let i = 0; i < W * H; i++){
    const x = i % W, p = i * 4;
    if (x < 30){ rgba[p] = 40; rgba[p + 1] = 160; rgba[p + 2] = 60; }
    else { rgba[p] = 150; rgba[p + 1] = 150; rgba[p + 2] = 150; }
    rgba[p + 3] = 255;
  }
  const m2 = new Uint8Array(W * H);
  const dolan = C.tasmaDoldur(rgba, gecerli, W, H, 10, 20, 25, m2, 1, 1e6);
  dogrula(dolan === 30 * H, 'değnek yalnız bitişik benzer alanı doldurur (' + dolan + '/' + 30 * H + ')');
  dogrula(m2[20 * W + 45] === 0, 'farklı renkli bölgeye taşmaz');
  const m3 = new Uint8Array(W * H);
  dogrula(C.tasmaDoldur(rgba, gecerli, W, H, 10, 20, 200, m3, 1, 1e6) === W * H,
    'yüksek tolerans tüm görüntüyü kapsar');
  gecerli[20 * W + 10] = 0;
  dogrula(C.tasmaDoldur(rgba, gecerli, W, H, 10, 20, 25, new Uint8Array(W * H), 1, 1e6) === 0,
    '"veri yok" tohumdan doldurma yapılmaz');
}

console.log('\n================================');
console.log('SONUÇ: ' + gecti + ' geçti, ' + kaldi + ' kaldı');
process.exit(kaldi ? 1 : 0);
