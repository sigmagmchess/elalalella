// -*- coding: utf-8 -*-
/**
 * YTÜ Harita Mühendisliği - Çam Ağacı Instance Segmentation
 * İlerleme Çubuğu Modülü (Python'daki tqdm kütüphanesinin karşılığı)
 *
 * Python sürümünde ilerleme çubuğu için tqdm kütüphanesi kullanılıyordu;
 * Node.js'te birebir karşılığı olmadığı için aynı görünümü veren bu küçük
 * modül yazılmıştır. Kullanımı tqdm ile aynıdır:
 *
 *   for (const eleman of ilerleme(dizi, "   Karolar işleniyor")) { ... }
 *
 * Yazar: YTÜ Harita Mühendisliği Yüksek Lisans Tezi
 * Tarih: 2024
 */

export class IlerlemeCubugu {
    /**
     * @param {number} toplam - Toplam adım sayısı
     * @param {string} aciklama - Çubuğun solunda gösterilecek açıklama (tqdm'deki desc)
     */
    constructor(toplam, aciklama = "") {
        this.toplam = toplam;
        this.aciklama = aciklama;
        this.sayac = 0;
        this.baslangic = Date.now();
        this.sonCizim = 0;
        this.ciz();
    }

    /** Sayacı n adım ilerletir ve çubuğu yeniler. */
    adim(n = 1) {
        this.sayac += n;
        const simdi = Date.now();
        // Konsolu boğmamak için en fazla 10 kez/saniye çiz (son adım hariç)
        if (simdi - this.sonCizim >= 100 || this.sayac >= this.toplam) {
            this.ciz();
            this.sonCizim = simdi;
        }
    }

    /** Çubuğu ekrana çizer (tqdm görünümü: açıklama: %42|████      | 42/100). */
    ciz() {
        const oran = this.toplam > 0 ? Math.min(this.sayac / this.toplam, 1) : 1;
        const genislik = 20;
        const dolu = Math.round(oran * genislik);
        const cubuk = "█".repeat(dolu) + " ".repeat(genislik - dolu);
        const gecen = (Date.now() - this.baslangic) / 1000;
        const hiz = gecen > 0 ? (this.sayac / gecen).toFixed(1) : "?";
        process.stdout.write(
            `\r${this.aciklama}: %${Math.round(oran * 100).toString().padStart(3)}` +
            `|${cubuk}| ${this.sayac}/${this.toplam} [${gecen.toFixed(0)}s, ${hiz} adet/s]`
        );
    }

    /** Çubuğu tamamlar ve satırı kapatır. */
    bitir() {
        this.sayac = this.toplam;
        this.ciz();
        process.stdout.write("\n");
    }
}

/**
 * tqdm(dizi, desc) kullanımının birebir karşılığı: diziyi dolaşırken
 * ilerleme çubuğunu otomatik günceller.
 *
 * @param {Array} dizi - Dolaşılacak dizi
 * @param {string} aciklama - Açıklama metni
 */
export function* ilerleme(dizi, aciklama = "") {
    const cubuk = new IlerlemeCubugu(dizi.length, aciklama);
    for (const eleman of dizi) {
        yield eleman;
        cubuk.adim();
    }
    cubuk.bitir();
}
