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
     Havuz listesinde her fotoğrafın yanında **✏️ Düzenle** (fotoğrafı
     tarayıcıdaki kayıtlı kopyasından kendiliğinden açar ve etiket vuruşlarını
     üzerine getirir; düzeltip yeniden eklersiniz — kopya yoksa dosya seçme
     penceresi açılır, doğru dosya seçilince vuruşlar yine kendiliğinden
     yerleşir) ve **✕ Çıkar** (yalnız o fotoğrafı siler, diğerleri korunur)
     bulunur. Vuruşlar havuz kaydıyla birlikte proje dosyasına da yazılır.
     Yüklenen her görüntünün çalışma boyutundaki kopyası (JPEG q=0,95,
     saydamlık varsa PNG; 20 MP üstü saydamlıklı ortomozaikler hariç) tarayıcıda
     saklanır; kopyadan açılan görüntü bilgi tablosunda "kayıtlı kopya" olarak
     görünür.
   - **Adım 4 → Taç içi boşluk doldurma:** çam taçları bütün alan olarak (dallar
     ve küçük boşluklar dahil) etiketlense bile model dal/gölge aralıklarını yer
     yer "Çam değil" diyebilir. Bu seçenek hedef sınıf maskesinde taç İÇİNDE
     kalan delikleri doldurur: görüntü kenarına değmeyen, veri-yok içermeyen ve
     alanı seçilen sınırı (≤0,5 / 2 / 5 m²) aşmayan arka plan adacıkları çam
     sayılır. Özgün hücreler silinmez ve komşu taçlar birleştirilmez (morfolojik
     kapamadan farkı budur; kapama denendi, kümeleri birleştirip tespiti
     bozduğu için kullanılmadı). Seyrek taçlı çamlar parçalanmaz, ağaç başına
     tek nokta kalır, taç poligonu referans taç gibi bütün çıkar. Ayar taraması
     ve rapor bu ayarı kullanır/yazar; proje dosyasında saklanır.
   - **Doğruluk Analizi → 🎓 Eğitim ağaçları (değerlendirme dışı):** eğitimde
     kullandığınız çamları nokta dosyasının `fid` kimlikleriyle (örn. `3,4,7,10-14`)
     ya da ayrı bir GeoJSON ile verin. Bu ağaçlar referanstan çıkarılır;
     taçlarına/yarıçaplarına düşen tespitler ne TP ne FP sayılır. Böylece F1,
     yanlış alarm ve kaçan ağaç sayıları **yalnız bağımsız (test) ağaçlar**
     üzerinden hesaplanır; rapor bunu ayrı bölümde belirtir. Taç poligonları
     dışlanan noktayı içermelerine göre dışlanır (katmanların fid'leri farklı
     olabilir). Referanslar tuvalde fid numaralarıyla çizilir; dışlananlar gri
     kesiklidir ("Referansları ve numaralarını tuvalde göster" kutusu).
     **Sınıf eklemek havuzu boşaltmaz** (yeni sınıf sona eklenir); sınıf
     silmek yalnız o sınıfın örneklerini havuzdan çıkarır, kalan kaynaklar durur.
     **Otomatik yedek:** havuz ve ekrandaki etiketler her değişiklikten ~1,5 s
     sonra tarayıcının IndexedDB'sine yazılır. Sekme kapansa ya da başka bir
     `kopru.html` kopyası açılsa bile aynı tarayıcıda sayfa açılınca Adım 1'de
     "Tarayıcı yedeği bulundu" notu çıkar; **⤵️ Geri yükle** ile havuz ve
     etiketler geri gelir (kaynak sayısı azalan bir işlemden önceki durum da
     "bir önceki yedek" olarak tutulur). Bu bir kaza sigortasıdır; proje
     dosyası (Adım 5) yerine geçmez ve başka bir tarayıcıya/bilgisayara taşınmaz.
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

## Doğruluk Analizi — referans verisi nasıl hazırlanır

Üç referans türü vardır; hepsi birlikte yüklenebilir ve **🎯 Değerlendir** her
biri için ayrı sonuç üretir:

| Referans | QGIS katmanı | Ölçüt |
|---|---|---|
| 🌿 **Taç poligonları** | Çokgen (Polygon/MultiPolygon), her taç bir öge | Alansal — IoU eşleştirme, alan R² |
| 📍 **Ağaç noktaları** | Nokta (Point), her ağaç bir öge | Noktasal — her ağaç için eşleşme yarıçapı içindeki en yakın tespit **bire bir** eşlenir; konum hatası (m) raporlanır |
| 🌳 **Küme poligonları** | Çokgen; sınırları ayrılamayan ağaç grupları | İçine ≥1 tespit düşen küme **bulundu** sayılır; kümedeki fazla tespit yanlış alarm sayılmaz; boş küme kaçırılmış sayılır |

Hepsini **ortomozaikle aynı EPSG'de** GeoJSON olarak dışa aktarın. Dördüncü yol:
**✍️ Elle değerlendir** — tuvalde işaretlersiniz, ayrı katmanda tutulur.

**Ağaç başına tek nokta:** tespit artık sabit yarıçaplı bastırma yerine *taç
bileşenleri + uzaklık dönüşümü tepe sayımı* ile yapılır. Tek yuvarlak taç tek
nokta verir (olasılık ağırlıklı merkez); birleşmiş taçlar boyun yerinden
ayrılır ve her parça bir ağaç olur (küme üyeleri kesikli ikinci halkayla
çizilir, GeoJSON'da `kume_id`/`kume_adet`). Taç boyu değişkenliğinden (2–12 m)
etkilenmez.

**🔎 En iyi ayarı bul:** güven eşiği (%35–80) × taç çapı (0,6–1,8×) ızgarasını
sunucuya gitmeden tarar, nokta-F1'i en yüksek ayarı uygular ve ilk 5'i gösterir.
Tez için dürüst kullanım: ayarı bir *doğrulama* alt kümesinde seçin, bağımsız
*test* alanında raporlayın — rapor bu notu kendiliğinden taşır.

Önemli ayrıntılar:
- Referans dosyası tüm sahayı kapsıyor ama yalnız bir parçasını analiz
  ediyorsanız sorun değil: **görüntü kapsamı dışındaki referanslar değerlendirmeye
  alınmaz** (kaç tanesinin elendiği arayüzde ve raporda yazar). Hiçbiri kapsama
  düşmüyorsa CRS uyuşmuyordur.
- **Ort. taç çapı**'nı gerçek ortalamanıza yakın girin (QGIS poligonlarının
  eşdeğer çapından); **eşleşme yarıçapı** ≈ taç yarıçapı (örn. 6 m taç → 3 m).
- IoU, sınıflandırma **bloğu** çözünürlüğünde hesaplanır. Çok büyük bir
  ortomozaikte işleme çözünürlüğü 3072 px kalırsa blok adımı metre mertebesine
  çıkar; taç düzeyinde IoU için **işleme çözünürlüğünü 6144/8192** yapın ya da
  sahanın bir **parçasını** analiz edin ve Tarama sıklığını **Sık** seçin.

Doğrulama (gerçek QGIS referansı, 2 cm/px çalışma çözünürlüğü, 13 tek taç + 3
küme): noktasal F1 **%96,8**, yanlış alarm 0, ortalama konum hatası **0,80 m**.

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
