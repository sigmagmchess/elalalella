#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Çam-AI ↔ YOLO köprüsü: karo bazlı YOLO tespitlerini ortomozaik koordinatına taşır.

Girdi
  • veri klasörü  : Çam-AI'nin ürettiği ZIP'in açılmış hâli (tahmin/karolar.json burada)
  • tahmin dosyası: [{"dosya": "karo_00000_00000.jpg", "kutular": [[x1,y1,x2,y2,guven], ...]}, ...]
                    (kutu köşeleri karo pikseli, 0..karo boyu)
Çıktı (cikti klasörü)
  • yolo_tespitler.geojson : ağaç başına TEK NOKTA (kutu merkezi) + "guven" özelliği → Çam-AI "Dış tespitleri yükle"
  • yolo_kutular.geojson   : aynı tespitlerin kutu poligonları (QGIS'te bakmak için)
  • yolo_ozet.txt          : eşik başına tespit sayısı

Yalnız standart kütüphane kullanır (numpy gerekmez); Colab defteri bu dosyayı gömülü olarak taşır.
"""
import argparse
import json
import math
import os


def karolari_oku(veri_klasoru):
    yol = os.path.join(veri_klasoru, 'tahmin', 'karolar.json')
    with open(yol, encoding='utf-8') as f:
        k = json.load(f)
    return k['meta'], {c['dosya']: c for c in k['karolar']}


def kutulari_calisma_pikseline_tasi(tahminler, karolar, karo_boyu):
    """Karo koordinatındaki kutuları çalışma (ortomozaik) pikseline çevirir; dolgu bölgesine
    düşenleri (karonun gerçek eninden/boyundan dışarıda kalan merkezler) atar."""
    liste = []
    for t in tahminler:
        karo = karolar.get(t['dosya']) or karolar.get(os.path.basename(t['dosya']))
        if karo is None:
            continue
        for k in t.get('kutular', []):
            if len(k) < 5:
                continue
            x1, y1, x2, y2, p = [float(v) for v in k[:5]]
            if x2 <= x1 or y2 <= y1:
                continue
            mx, my = (x1 + x2) / 2, (y1 + y2) / 2
            if mx > karo['w'] or my > karo['h']:          # siyah dolgu
                continue
            x1, x2 = max(0.0, x1), min(float(karo['w']), x2)
            y1, y2 = max(0.0, y1), min(float(karo['h']), y2)
            if x2 - x1 < 2 or y2 - y1 < 2:
                continue
            # karo kenarına değen kutu: ağaç karo sınırında kesilmiş olabilir (bindirme kopyası adayı)
            kenar = x1 <= 1.5 or y1 <= 1.5 or x2 >= karo['w'] - 1.5 or y2 >= karo['h'] - 1.5
            liste.append({'x1': karo['x0'] + x1, 'y1': karo['y0'] + y1,
                          'x2': karo['x0'] + x2, 'y2': karo['y0'] + y2,
                          'p': max(0.0, min(1.0, p)), 'karo': karo['dosya'], 'kenar': kenar})
    return liste


def _iou(a, b):
    kx = max(0.0, min(a['x2'], b['x2']) - max(a['x1'], b['x1']))
    ky = max(0.0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
    kes = kx * ky
    if kes <= 0:
        return 0.0
    aa = (a['x2'] - a['x1']) * (a['y2'] - a['y1'])
    ab = (b['x2'] - b['x1']) * (b['y2'] - b['y1'])
    return kes / (aa + ab - kes)


def _merkez_icinde(a, b):
    """b'nin merkezi a'nın içinde mi (bindirme sınırında yarım kalan kutular için)."""
    mx, my = (b['x1'] + b['x2']) / 2, (b['y1'] + b['y2']) / 2
    return a['x1'] <= mx <= a['x2'] and a['y1'] <= my <= a['y2']


def _kesisim_kucuk_orani(a, b):
    """Kesişim / küçük kutunun alanı (IoS). Karo sınırında ikiye bölünmüş aynı ağacın iki yarım kutusu
    IoU'da düşük kalır ama bindirme kuşağı kadar kesiştikleri için IoS yüksektir."""
    kx = max(0.0, min(a['x2'], b['x2']) - max(a['x1'], b['x1']))
    ky = max(0.0, min(a['y2'], b['y2']) - max(a['y1'], b['y1']))
    kes = kx * ky
    if kes <= 0:
        return 0.0
    kucuk = min((a['x2'] - a['x1']) * (a['y2'] - a['y1']), (b['x2'] - b['x1']) * (b['y2'] - b['y1']))
    return kes / kucuk if kucuk > 0 else 0.0


def kuresel_nms(kutular, iou_esik=0.45, ios_esik=0.5):
    """Güvene göre sıralı, tüm ortomozaik üzerinde bastırma. Bir kutu şu durumlarda elenir:
       • daha güvenli bir kutuyla IoU ≥ iou_esik (klasik NMS),
       • merkezi daha güvenli ve daha büyük bir kutunun içinde (karo kenarında yarım kalan kopya),
       • FARKLI karolardan gelen ve ikisi de karo kenarına değen iki kutunun kesişimi küçüğünün ≥ ios_esik'i
         (karo sınırında ikiye bölünmüş, bindirmeden büyük taç); tutulan kutu iki parçanın birleşimine
         genişletilir ki nokta taç merkezine düşsün. Aynı karodaki bitişik iki gerçek ağaç bu kurala girmez.
    Bindirmeli karolardan gelen kopyaları tek tespite indirger."""
    sirali = sorted(kutular, key=lambda k: -k['p'])
    tutulan = []
    for k in sirali:
        alan_k = (k['x2'] - k['x1']) * (k['y2'] - k['y1'])
        bastir = False
        for t in tutulan:
            if _iou(t, k) >= iou_esik:
                bastir = True
                break
            alan_t = (t['x2'] - t['x1']) * (t['y2'] - t['y1'])
            if alan_k <= alan_t and _merkez_icinde(t, k):
                bastir = True
                break
            if (t.get('kenar') and k.get('kenar') and t.get('karo') != k.get('karo')
                    and _kesisim_kucuk_orani(t, k) >= ios_esik):
                t['x1'], t['y1'] = min(t['x1'], k['x1']), min(t['y1'], k['y1'])
                t['x2'], t['y2'] = max(t['x2'], k['x2']), max(t['y2'], k['y2'])
                bastir = True
                break
        if not bastir:
            tutulan.append(dict(k))
    return tutulan


def cografi(meta, px, py):
    """Çalışma pikseli → (Doğu, Kuzey). Coğrafi bilgi yoksa özgün görüntü pikseli (y negatif) döner;
    Çam-AI bu biçimi de okur."""
    if meta.get('epsg') and meta.get('sx'):
        return meta['ox'] + px * meta['sx'], meta['oy'] - py * meta['sy']
    o = meta.get('olcek') or 1
    return px * o, -py * o


def geojson_yaz(kutular, meta, cikti_klasoru):
    os.makedirs(cikti_klasoru, exist_ok=True)
    crs = ({'type': 'name', 'properties': {'name': 'urn:ogc:def:crs:EPSG::%d' % int(meta['epsg'])}}
           if meta.get('epsg') else None)
    sx = meta.get('sx') or meta.get('olcek') or 1
    sy = meta.get('sy') or meta.get('olcek') or 1
    noktalar, poligonlar = [], []
    for i, k in enumerate(sorted(kutular, key=lambda k: -k['p']), start=1):
        mx, my = (k['x1'] + k['x2']) / 2, (k['y1'] + k['y2']) / 2
        e, n = cografi(meta, mx, my)
        oz = {'fid': i, 'guven': round(k['p'], 4), 'sinif': 'cam',
              'kutu_en': round((k['x2'] - k['x1']) * sx, 2), 'kutu_boy': round((k['y2'] - k['y1']) * sy, 2),
              'karo': k['karo']}
        noktalar.append({'type': 'Feature', 'properties': oz,
                         'geometry': {'type': 'Point', 'coordinates': [round(e, 3), round(n, 3)]}})
        kose = [cografi(meta, k['x1'], k['y1']), cografi(meta, k['x2'], k['y1']),
                cografi(meta, k['x2'], k['y2']), cografi(meta, k['x1'], k['y2'])]
        kose = [[round(a, 3), round(b, 3)] for a, b in kose]
        poligonlar.append({'type': 'Feature', 'properties': dict(oz),
                           'geometry': {'type': 'Polygon', 'coordinates': [kose + [kose[0]]]}})
    for ad, ozellikler in (('yolo_tespitler.geojson', noktalar), ('yolo_kutular.geojson', poligonlar)):
        fc = {'type': 'FeatureCollection', 'features': ozellikler}
        if crs:
            fc['crs'] = crs
        with open(os.path.join(cikti_klasoru, ad), 'w', encoding='utf-8') as f:
            json.dump(fc, f, ensure_ascii=False)
    satirlar = ['YOLO tespit özeti — kaynak: %s' % meta.get('kaynak', '?'),
                'Toplam (NMS sonrası): %d' % len(kutular), 'Güven eşiği → tespit sayısı:']
    for e in range(5, 100, 5):
        satirlar.append('  ≥ %%%2d : %d' % (e, sum(1 for k in kutular if k['p'] >= e / 100 - 1e-9)))
    with open(os.path.join(cikti_klasoru, 'yolo_ozet.txt'), 'w', encoding='utf-8') as f:
        f.write('\n'.join(satirlar) + '\n')
    return satirlar


def birlestir(veri_klasoru, tahminler, cikti_klasoru, nms_iou=0.45):
    """Tam akış: karo kutuları → çalışma pikseli → küresel NMS → GeoJSON. Tutulan kutuları döndürür."""
    meta, karolar = karolari_oku(veri_klasoru)
    ham = kutulari_calisma_pikseline_tasi(tahminler, karolar, meta.get('karo', 1024))
    tutulan = kuresel_nms(ham, nms_iou)
    ozet = geojson_yaz(tutulan, meta, cikti_klasoru)
    print('Ham kutu: %d → NMS sonrası: %d' % (len(ham), len(tutulan)))
    print('\n'.join(ozet))
    return tutulan


def main():
    ap = argparse.ArgumentParser(description='YOLO karo tespitlerini ortomozaik GeoJSON\'una çevirir')
    ap.add_argument('--veri', required=True, help='açılmış Çam-AI YOLO veri klasörü')
    ap.add_argument('--tahmin', required=True, help='karo tahminleri JSON')
    ap.add_argument('--cikti', required=True, help='çıktı klasörü')
    ap.add_argument('--nms', type=float, default=0.45, help='küresel NMS IoU eşiği')
    a = ap.parse_args()
    with open(a.tahmin, encoding='utf-8') as f:
        tahminler = json.load(f)
    birlestir(a.veri, tahminler, a.cikti, a.nms)


if __name__ == '__main__':
    main()
