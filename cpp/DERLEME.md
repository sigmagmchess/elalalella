# 🌲 Çam-AI C++ Eğitim Sunucusu — Derleme ve Kullanım

~10 milyon parametreli derin YSA'yı (36 öznitelik → 2096 → 2096 → 1568 → 1040 → 512 → N sınıf;
ReLU + BatchNorm + Dropout 0,2 + Adam + kosinüs öğrenme çizelgesi + erken durdurma, tohum=42)
bilgisayarınızda eğitir ve `http://localhost:8787` üzerinden HTML'e bağlar.
`kopru.html`, Çam-AI uygulamasının **tam kendisidir** (aynı arayüz: etiketleme
fırçası/çokgeni/değneği, sınıflandırma haritası, nokta atışı ağaç işaretleme,
taç poligonları, doğruluk analizi, tez raporu, tüm dışa aktarımlar) — tek farkla:
eğitim ve tahmin tarayıcıdaki küçük modeller yerine **bu C++ sunucusundaki
~10M parametreli derin YSA** ile yapılır.

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

## Kullanım

1. `cam_ai.exe` — çift tıklayın (ya da `cam_ai.exe 8787`). Konsolda
   "sunucu hazır" yazısını görün. Yalnızca `localhost` dinlenir; dışarıya kapalıdır.
2. `kopru.html` — tarayıcıda açın → **Adım 0: 🔌 Bağlan** (açılışta kendiliğinden de dener).
3. Sonrası bildiğiniz Çam-AI akışı:
   - **Adım 1** ortomozaiği yükleyin (GeoTIFF/JPG/PNG; EPSG + GSD otomatik okunur).
   - **Adım 2** fırça/çokgen/değnekle etiketleyin; ＋ Sınıf ekle ile
     istediğiniz kadar tür tanımlayın (Çam, Dişbudak, …). Çoklu eğitim için
     **➕ Örnekleri havuza ekle** ile birden çok görüntü birleştirilebilir.
   - **Adım 3** 🧠 Eğitimi Başlat: örnekler C++ sunucusuna gönderilir,
     ~10M parametreli derin YSA orada eğitilir — **canlı kayıp/F1 eğrisi**,
     **karışıklık matrisi** ve **tür bazlı Precision/Recall/F1** panelde;
     ⏹ Durdur ile erken kesilebilir (en iyi ağırlıklar korunur). Fisher
     paneli, verinizde keşfedilen ayırt edici kanalları göstermeye devam eder.
   - **Adım 4** 🗺️ sınıflandırma haritası (bloklar C++ modelinden tahmin edilir),
     **tür seçip** 📍 nokta atışı ağaç işaretleme, ⭐ en yoğun bölge,
     🌿 taç poligonları; **Doğruluk Analizi** (QGIS referans GeoJSON + IoU) ve
     **🔎 Analiz Et** (eğitim sonrası yeni harita bırak → otomatik işaretle) aynen çalışır.
   - **Adım 5** dışa aktarımlar: PNG · CSV · GeoJSON (EPSG etiketli) ·
     ağaç noktaları · taç poligonları · **C++ modelini indir (.bin)** ·
     proje kaydet/yükle · **🎓 Tez Raporu** (artık C++ eğitim sonuçları,
     mimari ve karışıklık matrisi bölümü de içerir).

Model her eğitim sonunda `cam_ai_model.bin` olarak diske kaydedilir; sonraki
açılışta **Adım 0 → "📥 Sunucudaki kayıtlı modeli kullan"** ile eğitimsiz devam
edersiniz.

## Dosyalar arasında geçiş — hangi dosya ne işe yarar

| Dosya | Nerede oluşur | Ne yapar |
|---|---|---|
| `*_proje.json` | Adım 5 → 💾 Projeyi kaydet | Etiketleriniz (fırça vuruşları), sınıflar, ayarlar ve **veri havuzunuz**. Yedek ve devam dosyanız budur. |
| `cam_ai_havuz.json` | `cam_ai.exe`'nin yanında, kendiliğinden | Sunucuda biriken ham eğitim örnekleri. **Adım 5 → 📥 Proje yükle** ile bu dosyayı da açabilirsiniz: örnekler uygulamanın havuzuna gelir, görüntü yüklemeden bile yeniden eğitebilirsiniz. |
| `cam_ai_model.bin` | `cam_ai.exe`'nin yanında, kendiliğinden | Eğitilmiş ağırlıklar (+ sınıf adları ve örnek penceresi). Açılışta otomatik yüklenir; taşımak için Adım 5'ten indirilebilir. |

**Proje dosyasını açma sırası önemsizdir:** görüntüden önce açarsanız uygulama
projeyi bekletir ve doğru görüntüyü yüklediğiniz anda etiketler kendiliğinden
yerleşir (Adım 1'de turuncu bir not bunu hatırlatır). Kaydedilen işleme
çözünürlüğü de geri yüklenir — etiketlerin piksel piksel oturması için bu şart.

Not: "Sınıf başına örnek" sınırı nedeniyle eğitime yalnızca bir altörneklem
gidiyorsa sunucudaki birikmiş havuza dokunulmaz (verileriniz korunur); tüm veri
gönderiliyorsa sunucu havuzu onunla eşitlenir. Havuzun tamamıyla eğitmek için
bu seçeneği **"Tümü"** yapın.

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
