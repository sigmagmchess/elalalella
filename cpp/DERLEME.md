# 🌲 Çam-AI C++ Eğitim Sunucusu — Derleme ve Kullanım

~10 milyon parametreli derin YSA'yı (36 öznitelik → 2096 → 2096 → 1568 → 1040 → 512 → N sınıf;
ReLU + BatchNorm + Dropout 0,2 + Adam + kosinüs öğrenme çizelgesi + erken durdurma, tohum=42)
bilgisayarınızda eğitir ve `http://localhost:8787` üzerinden HTML'e bağlar.
Köprü (`kopru.html`) yalnızca eğitim ekranı değildir: **5. adımda** bir ortomozaik
(GeoTIFF/JPG/PNG) bırakırsınız, modelin tanıdığı **her ağaç türü** görüntüde
kendi rengiyle nokta atışı işaretlenir (çok türlü tespit + CSV/GeoJSON/PNG dışa aktarım).

## Windows'ta .exe yapmak

**Seçenek A — MinGW-w64 (önerilen, ücretsiz):**
[winlibs.com](https://winlibs.com) üzerinden MinGW-w64 (UCRT, POSIX threads) indirin,
`bin` klasörünü PATH'e ekleyin, sonra:

```bat
g++ -O3 -march=native -fopenmp -static -o cam_ai.exe cam_ai_sunucu.cpp -lws2_32
```

**Seçenek B — Visual Studio (MSVC):**
"x64 Native Tools Command Prompt" açın:

```bat
cl /O2 /EHsc /openmp cam_ai_sunucu.cpp ws2_32.lib /Fe:cam_ai.exe
```

> `-fopenmp` / `/openmp` çok çekirdek kullanımı içindir; olmadan da derlenir
> (eğitim yalnızca yavaşlar). `-march=native` AVX hızlandırması sağlar.

Linux/macOS (test): `g++ -O3 -march=native -fopenmp -o cam_ai cam_ai_sunucu.cpp -lpthread`

## Kullanım (3 adım)

1. `cam_ai.exe` — çift tıklayın (ya da `cam_ai.exe 8787`). Konsolda
   "sunucu hazır" yazısını görün. Yalnızca `localhost` dinlenir; dışarıya kapalıdır.
2. `kopru.html` — tarayıcıda açın → **Bağlan**.
3. Çam-AI uygulamasında etiketleyin → **📚 Örnekleri havuza ekle** →
   **💾 Projeyi kaydet** → çıkan `*_proje.json` dosyasını köprüye bırakın →
   **🧠 Eğitimi Başlat**. Canlı kayıp/F1 grafiği köprüde; ⏹ Durdur ile
   erken kesebilirsiniz (en iyi ağırlıklar korunur). Bitince sahnede
   **karışıklık matrisi** ve **tür bazlı Precision/Recall/F1** tablosu görünür;
   **🎓 Tez raporu (TXT)** düğmesi tarihli, tekrarlanabilir (tohum=42) bir eğitim
   raporu indirir. **💾 Modeli indir** (`cam_ai_10m.bin`, ~38 MB) yedek almak içindir —
   model zaten diske otomatik kaydedilir.

## 5. Adım — Görüntü Analizi (çok türlü nokta atışı tespit)

Eğitimden (ya da kayıtlı model yüklendikten) sonra **5. adıma** ortomozaik bırakın:

- **GeoTIFF/TIFF** doğrudan tarayıcıda çözülür (LZW/Deflate/PackBits/JPEG,
  BigTIFF, şerit/karo, nodata); JPG/PNG de olur. Koordinat sistemi (EPSG) ve
  GSD otomatik okunur.
- Görüntü, eğitimdekiyle aynı **36 öznitelikli** bloklara bölünür; bloklar
  parça parça sunucuya gönderilir (`POST /tahmin`) — büyük görüntülerde
  ilerleme çubuğu ve ⏹ durdurma vardır.
- Modelin tanıdığı **her tür için** ayrı renkte nokta atışı işaret (NMS +
  olasılık ağırlıklı merkez inceltme), tür onay kutularıyla filtre,
  güven eşiği ve ortalama taç çapı ayarı, ⭐ en yoğun bölge.
- **Örnek penceresi** (eğitim örneği boyu, px) proje dosyasından ya da
  sunucudaki kalıcı havuzdan otomatik gelir (`/egit` gövdesindeki `yama`
  alanı havuzda saklanır, `/durum` içinde geri döner).
- Dışa aktarım: **CSV** (tür, güven, piksel + UTM koordinatları),
  **GeoJSON** (QGIS'te açılır, EPSG etiketli, tür öznitelikli),
  **işaretli PNG**.

> Eşik ve taç çapı değişiklikleri sunucuya gitmeden anında yeniden
> işaretlenir; yalnızca "Analiz Et" sunucuda tahmin çalıştırır.

## Kendi HTML'inizden bağlanmak

CORS açıktır; herhangi bir sayfadan `fetch` yeterli:

```js
// eğitim durumu
const d = await (await fetch('http://localhost:8787/durum')).json();

// tahmin (ham 36'lık öznitelik vektörleri, m×d düz dizi)
const y = await (await fetch('http://localhost:8787/tahmin', {
  method: 'POST', headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({ X: ozellikDizisi })
})).json();   // → { probs: [[K olasılık]...], sinif: [m] }
```

Uçlar: `GET /durum` · `POST /egit` · `POST /durdur` · `POST /tahmin` ·
`GET /model` (bin indir) · `POST /model` (bin yükle) ·
`GET /havuz` (kayıtlı havuz özeti) · `POST /havuz/bosalt` ·
`GET /rapor` (son eğitimin tam tez raporu: ayarlar, epoch geçmişi,
karışıklık matrisi, tür bazlı P/R/F1, makro-F1, süre — JSON).

## Kalıcı öğrenme (yeni)

Sunucu, öğrendiklerini **exe'nin yanındaki iki dosyaya** kaydeder ve her
açılışta otomatik geri yükler:

| Dosya | İçerik |
|---|---|
| `cam_ai_havuz.json` | Birikimli eğitim verisi — `/egit`'e gönderilen her yeni veri, **sınıf adına göre** öncekilerle birleştirilir; örnek penceresi (`yama`) de burada saklanır |
| `cam_ai_model.bin` | En son eğitilen model (sınıf adları dahil) — açılışta yüklenir, `/tahmin` hemen çalışır |

`/egit` varsayılan olarak **tüm havuzla** eğitir (yeni gönderilen veri önce
havuza eklenir). Yalnızca gönderdiğiniz veriyle eğitmek için gövdeye
`"sadeceBu": true` ekleyin.

`POST /egit` gövdesi:

```json
{ "d": 36, "K": 3, "siniflar": ["Çam","Dişbudak","Ağaç Değil"],
  "X": [n*d sayı], "y": [n adet 0..K-1],
  "ayarlar": { "hedefParam": 10000000, "enCokDevir": 120,
               "parti": 128, "hiz": 0.001, "dropout": 0.2, "sabir": 12 } }
```

## Notlar

- Öznitelikler **ham** gönderilir (standartlaştırma gerekmez; ilk katmandaki
  BatchNorm bunu içeride halleder). Çam-AI proje dosyasındaki havuz örnekleri
  zaten bu biçimdedir.
- Model dosyası biçimi: `CAMAI10M` imzalı, katman boyutları + float32 ağırlıklar.
- Tekrarlanabilirlik: ağırlık başlatma ve veri karıştırma tohum=42.
- Bellek: ~10M parametre için eğitim sırasında ≈ 300–400 MB RAM.
- Bu sunucu, tezdeki tarayıcı-içi modele (36 öznitelik uzayı aynı) **büyük model
  karşılaştırması** eklemenizi sağlar; sınıflandırma haritası üretimi için
  Çam-AI'nin blok özniteliklerini `/tahmin` ucuna göndermek yeterlidir.
