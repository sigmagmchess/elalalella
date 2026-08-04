// =====================================================================
// ÇAM-AI YEREL EĞİTİM SUNUCUSU (C++17, tek dosya, harici bağımlılık yok)
// ---------------------------------------------------------------------
// ~10 milyon parametreli derin YSA (MLP: 36 öznitelik → ... → N sınıf,
// ReLU + BatchNorm + Dropout + Adam + kosinüs öğrenme çizelgesi + erken
// durdurma, tohum=42) ve onu tarayıcıya bağlayan küçük bir HTTP sunucusu.
//
// Derleme (Windows, MinGW-w64):
//   g++ -O3 -march=native -fopenmp -static -o cam_ai.exe cam_ai_sunucu.cpp -lws2_32
// Derleme (Windows, MSVC "x64 Native Tools" komut istemi):
//   cl /O2 /EHsc /openmp cam_ai_sunucu.cpp ws2_32.lib /Fe:cam_ai.exe
// Derleme (Linux/macOS, test için):
//   g++ -O3 -march=native -fopenmp -o cam_ai cam_ai_sunucu.cpp -lpthread
//
// Çalıştırma:  cam_ai.exe [port]      (varsayılan 8787)
// Ardından kopru.html'yi tarayıcıda açın (ya da kendi HTML'inizden
// http://localhost:8787 adresine fetch ile bağlanın; CORS açıktır).
//
// HTTP API (tümü JSON, UTF-8):
//   GET  /durum            → sunucu + eğitim durumu (epoch, kayıp, valF1...)
//   POST /egit             → {d,K,siniflar:[..],X:[n*d sayı],y:[n],ayarlar:{...}}
//                            eğitimi arka plan iş parçacığında başlatır
//   POST /durdur           → eğitimi nazikçe durdurur (en iyi ağırlıklar kalır)
//   POST /tahmin           → {X:[m*d]} → {probs:[[K]...], sinif:[m]}
//   GET  /model            → eğitilmiş modeli ikili (.bin) indirir
//   POST /model            → ikili model dosyasını yükler (gövde = .bin içeriği)
// =====================================================================

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <string>
#include <vector>
#include <map>
#include <atomic>
#include <mutex>
#include <thread>
#include <random>
#include <algorithm>
#include <sstream>
#include <fstream>

// Öğrenilenler exe'nin yanındaki bu dosyalara kalıcı yazılır:
static const char* HAVUZ_DOSYA = "cam_ai_havuz.json";   // birikimli eğitim verisi
static const char* MODEL_DOSYA = "cam_ai_model.bin";    // en son eğitilen model

#ifdef _WIN32
  #include <winsock2.h>
  #include <ws2tcpip.h>
  typedef SOCKET soket_t;
  #define SOKET_KAPAT closesocket
#else
  #include <sys/socket.h>
  #include <netinet/in.h>
  #include <arpa/inet.h>
  #include <unistd.h>
  typedef int soket_t;
  #define SOKET_KAPAT close
  #define INVALID_SOCKET (-1)
#endif

// ============================ küçük JSON ==============================
// Yalnızca bu uygulamanın ihtiyaç duyduğu alt küme: nesne, dizi, sayı,
// dizgi (\u kaçışları dahil), true/false/null.
struct Json {
  enum Tip { NUL, SAYI, DIZGI, DIZI, NESNE, BOOL } tip = NUL;
  double sayi = 0;
  bool dogru = false;
  std::string dizgi;
  std::vector<Json> dizi;
  std::map<std::string, Json> nesne;

  const Json* al(const std::string& k) const {
    auto it = nesne.find(k);
    return it == nesne.end() ? nullptr : &it->second;
  }
  double sayiAl(const std::string& k, double varsayilan) const {
    const Json* j = al(k);
    return (j && j->tip == SAYI) ? j->sayi : varsayilan;
  }
};

struct JsonAyristirici {
  const char* p; const char* son; bool hata = false;
  JsonAyristirici(const std::string& s) : p(s.data()), son(s.data() + s.size()) {}
  void bosluk(){ while (p < son && (*p==' '||*p=='\t'||*p=='\n'||*p=='\r')) p++; }
  Json coz(){ bosluk(); if (p >= son){ hata = true; return {}; }
    char c = *p;
    if (c=='{') return nesneCoz();
    if (c=='[') return diziCoz();
    if (c=='"') return dizgiCoz();
    if (c=='t'||c=='f') return boolCoz();
    if (c=='n'){ p += 4; return {}; }
    return sayiCoz();
  }
  Json nesneCoz(){ Json j; j.tip = Json::NESNE; p++;
    bosluk();
    if (p < son && *p=='}'){ p++; return j; }
    while (p < son){
      bosluk();
      Json anahtar = dizgiCoz();
      bosluk();
      if (p >= son || *p != ':'){ hata = true; return j; }
      p++;
      j.nesne[anahtar.dizgi] = coz();
      bosluk();
      if (p < son && *p==','){ p++; continue; }
      if (p < son && *p=='}'){ p++; return j; }
      hata = true; return j;
    }
    hata = true; return j;
  }
  Json diziCoz(){ Json j; j.tip = Json::DIZI; p++;
    bosluk();
    if (p < son && *p==']'){ p++; return j; }
    while (p < son){
      j.dizi.push_back(coz());
      bosluk();
      if (p < son && *p==','){ p++; continue; }
      if (p < son && *p==']'){ p++; return j; }
      hata = true; return j;
    }
    hata = true; return j;
  }
  Json dizgiCoz(){ Json j; j.tip = Json::DIZGI;
    if (p >= son || *p != '"'){ hata = true; return j; }
    p++;
    while (p < son && *p != '"'){
      if (*p == '\\' && p + 1 < son){
        p++;
        char c = *p++;
        switch (c){
          case 'n': j.dizgi += '\n'; break;
          case 't': j.dizgi += '\t'; break;
          case 'r': j.dizgi += '\r'; break;
          case 'b': case 'f': break;
          case 'u': {
            if (p + 4 <= son){
              unsigned kod = (unsigned)strtoul(std::string(p, p+4).c_str(), nullptr, 16);
              p += 4;
              // UTF-8'e çevir (temel düzlem yeterli)
              if (kod < 0x80) j.dizgi += (char)kod;
              else if (kod < 0x800){
                j.dizgi += (char)(0xC0 | (kod >> 6));
                j.dizgi += (char)(0x80 | (kod & 0x3F));
              } else {
                j.dizgi += (char)(0xE0 | (kod >> 12));
                j.dizgi += (char)(0x80 | ((kod >> 6) & 0x3F));
                j.dizgi += (char)(0x80 | (kod & 0x3F));
              }
            }
            break;
          }
          default: j.dizgi += c;
        }
      } else j.dizgi += *p++;
    }
    if (p < son) p++;
    return j;
  }
  Json boolCoz(){ Json j; j.tip = Json::BOOL;
    if (*p=='t'){ j.dogru = true; p += 4; } else { j.dogru = false; p += 5; }
    return j;
  }
  Json sayiCoz(){ Json j; j.tip = Json::SAYI;
    char* uc = nullptr;
    j.sayi = strtod(p, &uc);
    if (uc == p){ hata = true; } else p = uc;
    return j;
  }
};

static std::string jsonKacis(const std::string& s){
  std::string c;
  for (unsigned char ch : s){
    if (ch=='"') c += "\\\"";
    else if (ch=='\\') c += "\\\\";
    else if (ch=='\n') c += "\\n";
    else if (ch < 0x20) { char t[8]; snprintf(t, 8, "\\u%04x", ch); c += t; }
    else c += (char)ch;
  }
  return c;
}

// ============================ derin YSA ===============================
struct DerinYsa {
  int d = 0, K = 0;
  std::vector<int> boyut;                 // d, h1..h4, K
  long long parametre = 0;
  // katman parametreleri (float — hız ve bellek)
  std::vector<std::vector<float>> W;      // gizli: out×in
  std::vector<std::vector<float>> gamma, beta, runM, runV;
  std::vector<float> Ws, bs;              // çıkış katmanı

  static std::vector<int> boyutSec(int d, int K, long long hedef){
    const int taban[4] = {2048, 2048, 1536, 1024};
    std::vector<int> enIyi;
    long long enFark = -1;
    for (double s = 0.5; s <= 2.0001; s += 0.01){
      int h[5];
      for (int i = 0; i < 4; i++) h[i] = std::max(32, (int)std::lround(taban[i] * s / 16) * 16);
      h[4] = std::max(64, h[3] / 2 / 16 * 16);          // beşinci gizli katman
      long long p = (long long)d * h[0];
      for (int i = 0; i < 4; i++) p += (long long)h[i] * h[i + 1];
      p += (long long)h[4] * K + K;
      for (int i = 0; i < 5; i++) p += 2LL * h[i];
      long long fark = llabs(p - hedef);
      if (enFark < 0 || fark < enFark){
        enFark = fark;
        enIyi = { d, h[0], h[1], h[2], h[3], h[4], K };
      }
    }
    return enIyi;
  }

  void kur(int d_, int K_, long long hedefParam){
    d = d_; K = K_;
    boyut = boyutSec(d, K, hedefParam);
    int L = (int)boyut.size() - 2;
    std::mt19937 rng(42);
    std::normal_distribution<float> N01(0.f, 1.f);
    W.assign(L, {}); gamma.assign(L, {}); beta.assign(L, {});
    runM.assign(L, {}); runV.assign(L, {});
    parametre = 0;
    for (int l = 0; l < L; l++){
      int nin = boyut[l], nout = boyut[l + 1];
      W[l].resize((size_t)nout * nin);
      float olc = std::sqrt(2.f / nin);
      for (auto& w : W[l]) w = N01(rng) * olc;
      gamma[l].assign(nout, 1.f);
      beta[l].assign(nout, 0.f);
      runM[l].assign(nout, 0.f);
      runV[l].assign(nout, 1.f);
      parametre += (long long)nout * nin + 2LL * nout;
    }
    int nin = boyut[L];
    Ws.resize((size_t)K * nin);
    float olc = std::sqrt(2.f / nin);
    for (auto& w : Ws) w = N01(rng) * olc;
    bs.assign(K, 0.f);
    parametre += (long long)K * nin + K;
  }

  // tek örnek çıkarımı (koşan istatistikler, dropout yok)
  void tahmin(const float* x, float* probs) const {
    int L = (int)boyut.size() - 2;
    static thread_local std::vector<float> a1, a2;
    a1.assign(x, x + d);
    std::vector<float>* girdi = &a1;
    std::vector<float>* cikti = &a2;
    for (int l = 0; l < L; l++){
      int nin = boyut[l], nout = boyut[l + 1];
      cikti->assign(nout, 0.f);
      const float* g = girdi->data();
      #pragma omp parallel for schedule(static)
      for (int o = 0; o < nout; o++){
        const float* w = &W[l][(size_t)o * nin];
        float z = 0;
        for (int j = 0; j < nin; j++) z += w[j] * g[j];
        float zn = (z - runM[l][o]) / std::sqrt(runV[l][o] + 1e-5f);
        float a = gamma[l][o] * zn + beta[l][o];
        (*cikti)[o] = a > 0 ? a : 0;
      }
      std::swap(girdi, cikti);
    }
    int nin = boyut[L];
    float enB = -1e30f;
    const float* g = girdi->data();
    for (int c = 0; c < K; c++){
      const float* w = &Ws[(size_t)c * nin];
      float z = bs[c];
      for (int j = 0; j < nin; j++) z += w[j] * g[j];
      probs[c] = z;
      if (z > enB) enB = z;
    }
    float top = 0;
    for (int c = 0; c < K; c++){ probs[c] = std::exp(probs[c] - enB); top += probs[c]; }
    for (int c = 0; c < K; c++) probs[c] /= top;
  }

  // v2 (CAMAI11M): sınıf adları da dosyada saklanır; v1 (CAMAI10M) okunmaya devam eder
  bool kaydet(std::vector<uint8_t>& cikti, const std::vector<std::string>& siniflar) const {
    if (boyut.empty()) return false;
    auto yaz32 = [&](int32_t v){ cikti.insert(cikti.end(), (uint8_t*)&v, (uint8_t*)&v + 4); };
    auto yazF = [&](const std::vector<float>& v){
      cikti.insert(cikti.end(), (const uint8_t*)v.data(), (const uint8_t*)v.data() + v.size() * 4);
    };
    cikti.clear();
    const char* imza = "CAMAI11M";
    cikti.insert(cikti.end(), imza, imza + 8);
    yaz32((int32_t)boyut.size());
    for (int b : boyut) yaz32(b);
    yaz32((int32_t)siniflar.size());
    for (const auto& s : siniflar){
      yaz32((int32_t)s.size());
      cikti.insert(cikti.end(), s.begin(), s.end());
    }
    int L = (int)boyut.size() - 2;
    for (int l = 0; l < L; l++){ yazF(W[l]); yazF(gamma[l]); yazF(beta[l]); yazF(runM[l]); yazF(runV[l]); }
    yazF(Ws); yazF(bs);
    return true;
  }
  bool yukle(const uint8_t* veri, size_t boy, std::vector<std::string>* siniflarCikti = nullptr){
    bool v2 = boy >= 12 && memcmp(veri, "CAMAI11M", 8) == 0;
    if (!v2 && (boy < 12 || memcmp(veri, "CAMAI10M", 8) != 0)) return false;
    size_t p = 8;
    auto oku32 = [&]() -> int32_t { int32_t v; memcpy(&v, veri + p, 4); p += 4; return v; };
    int nb = oku32();
    if (nb < 3 || nb > 16) return false;
    boyut.resize(nb);
    for (int i = 0; i < nb; i++) boyut[i] = oku32();
    d = boyut.front(); K = boyut.back();
    if (v2){
      int adSayi = oku32();
      if (adSayi < 0 || adSayi > 64) return false;
      if (siniflarCikti) siniflarCikti->clear();
      for (int i = 0; i < adSayi; i++){
        int uz = oku32();
        if (uz < 0 || p + (size_t)uz > boy) return false;
        std::string ad((const char*)veri + p, uz);
        p += uz;
        if (siniflarCikti) siniflarCikti->push_back(ad);
      }
    }
    int L = nb - 2;
    auto okuF = [&](std::vector<float>& v, size_t adet) -> bool {
      if (p + adet * 4 > boy) return false;
      v.resize(adet);
      memcpy(v.data(), veri + p, adet * 4);
      p += adet * 4;
      return true;
    };
    W.assign(L, {}); gamma.assign(L, {}); beta.assign(L, {}); runM.assign(L, {}); runV.assign(L, {});
    parametre = 0;
    for (int l = 0; l < L; l++){
      size_t nin = boyut[l], nout = boyut[l + 1];
      if (!okuF(W[l], nout * nin) || !okuF(gamma[l], nout) || !okuF(beta[l], nout) ||
          !okuF(runM[l], nout) || !okuF(runV[l], nout)) return false;
      parametre += (long long)(nout * nin) + 2LL * nout;
    }
    if (!okuF(Ws, (size_t)K * boyut[L]) || !okuF(bs, K)) return false;
    parametre += (long long)K * boyut[L] + K;
    return true;
  }
};

// ============================ eğitim durumu ===========================
struct Havuz {                            // bilgisayarda birikimli eğitim verisi
  int d = 0;
  std::vector<std::string> siniflar;
  std::vector<float> X;
  std::vector<int> y;
  std::vector<long long> sayimlar() const {
    std::vector<long long> s(siniflar.size(), 0);
    for (int c : y) if (c >= 0 && c < (int)s.size()) s[c]++;
    return s;
  }
};

struct Durum {
  std::atomic<bool> egitimde{false}, durdurIstek{false}, modelHazir{false};
  std::atomic<bool> modelDosyadan{false};
  std::atomic<int> epoch{0}, enCokDevir{0}, enIyiEpoch{0};
  std::atomic<double> kayip{0}, valKayip{0}, valF1{0}, enIyiF1{0}, lr{0};
  std::atomic<long long> parametre{0};
  std::mutex kilit;                      // model + gecmis + siniflar + havuz erişimi
  DerinYsa model;
  std::vector<std::string> siniflar;
  Havuz havuz;
  std::string gecmisJson = "[]";
  std::string sonHata;
  int nE = 0, nV = 0;
} G;

// ---- havuz kalıcılığı (kilit çağıran tarafta tutulur) ----
static void havuzKaydet(){
  std::ofstream f(HAVUZ_DOSYA, std::ios::binary);
  if (!f) return;
  f << "{\"d\":" << G.havuz.d << ",\"siniflar\":[";
  for (size_t i = 0; i < G.havuz.siniflar.size(); i++)
    f << (i ? "," : "") << '"' << jsonKacis(G.havuz.siniflar[i]) << '"';
  f << "],\"y\":[";
  for (size_t i = 0; i < G.havuz.y.size(); i++)
    f << (i ? "," : "") << G.havuz.y[i];
  f << "],\"X\":[";
  char sayi[32];
  for (size_t i = 0; i < G.havuz.X.size(); i++){
    snprintf(sayi, sizeof(sayi), "%.6g", (double)G.havuz.X[i]);
    f << (i ? "," : "") << sayi;
  }
  f << "]}";
}
static void havuzYukle(){
  std::ifstream f(HAVUZ_DOSYA, std::ios::binary);
  if (!f) return;
  std::string icerik((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
  JsonAyristirici ja(icerik);
  Json j = ja.coz();
  const Json* jX = j.al("X");
  const Json* jy = j.al("y");
  const Json* js = j.al("siniflar");
  if (ja.hata || !jX || !jy || !js) return;
  G.havuz.d = (int)j.sayiAl("d", 36);
  G.havuz.siniflar.clear();
  for (auto& sj : js->dizi) G.havuz.siniflar.push_back(sj.dizgi);
  G.havuz.y.clear();
  for (auto& v : jy->dizi) G.havuz.y.push_back((int)v.sayi);
  G.havuz.X.resize(jX->dizi.size());
  for (size_t i = 0; i < jX->dizi.size(); i++) G.havuz.X[i] = (float)jX->dizi[i].sayi;
  if (G.havuz.X.size() != G.havuz.y.size() * (size_t)G.havuz.d){
    G.havuz = Havuz{};                    // bozuk dosya — sıfırla
  }
}
// yeni örnekleri sınıf ADINA göre birleştirerek havuza ekler; eklenen örnek sayısını döndürür
static int havuzaEkle(int d, const std::vector<std::string>& adlar,
                      const std::vector<float>& X, const std::vector<int>& y){
  if (G.havuz.d == 0) G.havuz.d = d;
  if (G.havuz.d != d) return -1;          // öznitelik sayısı uyuşmalı
  std::vector<int> esle(adlar.size());
  for (size_t i = 0; i < adlar.size(); i++){
    auto it = std::find(G.havuz.siniflar.begin(), G.havuz.siniflar.end(), adlar[i]);
    if (it == G.havuz.siniflar.end()){
      G.havuz.siniflar.push_back(adlar[i]);
      esle[i] = (int)G.havuz.siniflar.size() - 1;
    } else esle[i] = (int)(it - G.havuz.siniflar.begin());
  }
  for (size_t i = 0; i < y.size(); i++){
    G.havuz.y.push_back(esle[y[i]]);
    G.havuz.X.insert(G.havuz.X.end(), X.begin() + i * d, X.begin() + (i + 1) * d);
  }
  havuzKaydet();
  return (int)y.size();
}
static void modelDosyayaKaydet(){
  std::vector<uint8_t> ikili;
  if (!G.model.kaydet(ikili, G.siniflar)) return;
  std::ofstream f(MODEL_DOSYA, std::ios::binary);
  if (f) f.write((const char*)ikili.data(), ikili.size());
}

static double makroF1(const std::vector<long long>& M, int K){
  double toplam = 0;
  for (int c = 0; c < K; c++){
    long long tp = M[(size_t)c * K + c], fp = 0, fn = 0;
    for (int c2 = 0; c2 < K; c2++){
      if (c2 == c) continue;
      fp += M[(size_t)c2 * K + c];
      fn += M[(size_t)c * K + c2];
    }
    double p = tp + fp > 0 ? (double)tp / (tp + fp) : 0;
    double r = tp + fn > 0 ? (double)tp / (tp + fn) : 0;
    toplam += p + r > 0 ? 2 * p * r / (p + r) : 0;
  }
  return toplam / K;
}

// ---------------------------------------------------------------------
// Eğitim: mini-batch, BN (parti istatistikleri) + dropout + Adam.
// Ana matematik, HTML içindeki JS derin YSA ile birebir aynıdır.
// ---------------------------------------------------------------------
static void egitimCalistir(std::vector<float> X, std::vector<int> y, int n, int d, int K,
                           long long hedefParam, int enCokDevir, float hiz, float l2,
                           int parti, float dropout, int sabir){
  std::mt19937 rng(42);
  std::mt19937 rngDrop(1042);
  {
    std::lock_guard<std::mutex> kilit(G.kilit);
    G.model.kur(d, K, hedefParam);
    G.parametre = G.model.parametre;
    G.gecmisJson = "[";
  }
  DerinYsa& M = G.model;
  int L = (int)M.boyut.size() - 2;

  // katmanlı %15 doğrulama bölmesi
  std::vector<std::vector<int>> gruplar(K);
  for (int i = 0; i < n; i++) gruplar[y[i]].push_back(i);
  std::vector<int> egit, val;
  for (auto& g : gruplar){
    std::shuffle(g.begin(), g.end(), rng);
    int vN = std::max(1, (int)std::lround(g.size() * 0.15));
    for (size_t i = 0; i < g.size(); i++)
      ((int)i < vN && g.size() > 3 ? val : egit).push_back(g[i]);
  }
  std::sort(egit.begin(), egit.end());
  std::sort(val.begin(), val.end());
  G.nE = (int)egit.size(); G.nV = (int)val.size();

  // Adam durumları
  auto adamY = [](size_t nAdet){ return std::vector<float>(nAdet, 0.f); };
  std::vector<std::vector<float>> mW(L), vW(L), mG(L), vG(L), mB(L), vB(L);
  for (int l = 0; l < L; l++){
    mW[l] = adamY(M.W[l].size()); vW[l] = adamY(M.W[l].size());
    mG[l] = adamY(M.gamma[l].size()); vG[l] = adamY(M.gamma[l].size());
    mB[l] = adamY(M.beta[l].size()); vB[l] = adamY(M.beta[l].size());
  }
  std::vector<float> mWs = adamY(M.Ws.size()), vWs = adamY(M.Ws.size());
  std::vector<float> mBs = adamY(M.bs.size()), vBs = adamY(M.bs.size());
  const float B1 = 0.9f, B2 = 0.999f, EPS = 1e-8f, BNE = 1e-5f;
  long long adimSay = 0;

  // parti tamponları
  std::vector<std::vector<float>> Z(L), ZN(L), A(L);
  std::vector<std::vector<uint8_t>> MSK(L);
  for (int l = 0; l < L; l++){
    size_t boyu = (size_t)parti * M.boyut[l + 1];
    Z[l].resize(boyu); ZN[l].resize(boyu); A[l].resize(boyu); MSK[l].resize(boyu);
  }
  std::vector<float> A0((size_t)parti * d), P((size_t)parti * K);
  std::vector<float> dUst, dAlt;
  std::vector<float> muB, varB;

  auto ileri = [&](const int* idx, int m, bool egitimModu){
    for (int s = 0; s < m; s++)
      memcpy(&A0[(size_t)s * d], &X[(size_t)idx[s] * d], d * 4);
    const float* girdi = A0.data();
    int nin = d;
    for (int l = 0; l < L; l++){
      int nout = M.boyut[l + 1];
      float* Zl = Z[l].data();
      #pragma omp parallel for schedule(static)
      for (int o = 0; o < nout; o++){
        const float* w = &M.W[l][(size_t)o * nin];
        for (int s = 0; s < m; s++){
          const float* g = girdi + (size_t)s * nin;
          float z = 0;
          for (int j = 0; j < nin; j++) z += w[j] * g[j];
          Zl[(size_t)s * nout + o] = z;
        }
      }
      muB.assign(nout, 0.f); varB.assign(nout, 0.f);
      float* ZNl = ZN[l].data();
      float* Al = A[l].data();
      uint8_t* Ml = MSK[l].data();
      if (egitimModu){
        for (int o = 0; o < nout; o++){
          float mu = 0;
          for (int s = 0; s < m; s++) mu += Zl[(size_t)s * nout + o];
          mu /= m;
          float va = 0;
          for (int s = 0; s < m; s++){ float f = Zl[(size_t)s * nout + o] - mu; va += f * f; }
          va /= m;
          muB[o] = mu; varB[o] = va;
          M.runM[l][o] = 0.9f * M.runM[l][o] + 0.1f * mu;
          M.runV[l][o] = 0.9f * M.runV[l][o] + 0.1f * va;
        }
      }
      const float* mk = egitimModu ? muB.data() : M.runM[l].data();
      const float* vk = egitimModu ? varB.data() : M.runV[l].data();
      float tut = 1.f - dropout;
      for (int s = 0; s < m; s++){
        size_t t0 = (size_t)s * nout;
        for (int o = 0; o < nout; o++){
          float zn = (Zl[t0 + o] - mk[o]) / std::sqrt(vk[o] + BNE);
          ZNl[t0 + o] = zn;
          float a = M.gamma[l][o] * zn + M.beta[l][o];
          a = a > 0 ? a : 0;
          if (egitimModu){
            uint8_t kal = (rngDrop() / (double)rngDrop.max()) < tut ? 1 : 0;
            Ml[t0 + o] = kal;
            a = kal ? a / tut : 0;
          }
          Al[t0 + o] = a;
        }
      }
      girdi = Al; nin = nout;
    }
    // çıkış + softmax
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < m; s++){
      const float* g = girdi + (size_t)s * nin;
      float enB = -1e30f;
      for (int c = 0; c < K; c++){
        const float* w = &M.Ws[(size_t)c * nin];
        float z = M.bs[c];
        for (int j = 0; j < nin; j++) z += w[j] * g[j];
        P[(size_t)s * K + c] = z;
        if (z > enB) enB = z;
      }
      float top = 0;
      for (int c = 0; c < K; c++){
        P[(size_t)s * K + c] = std::exp(P[(size_t)s * K + c] - enB);
        top += P[(size_t)s * K + c];
      }
      for (int c = 0; c < K; c++) P[(size_t)s * K + c] /= top;
    }
  };

  auto adam = [&](std::vector<float>& par, std::vector<float>& grad,
                  std::vector<float>& mm, std::vector<float>& vv, float lr){
    float d1 = 1.f - std::pow(B1, (float)adimSay), d2 = 1.f - std::pow(B2, (float)adimSay);
    #pragma omp parallel for schedule(static)
    for (long long i = 0; i < (long long)par.size(); i++){
      mm[i] = B1 * mm[i] + (1 - B1) * grad[i];
      vv[i] = B2 * vv[i] + (1 - B2) * grad[i] * grad[i];
      par[i] -= lr * (mm[i] / d1) / (std::sqrt(vv[i] / d2) + EPS);
    }
  };

  std::vector<float> gWs(M.Ws.size()), gBs(K);
  std::vector<std::vector<float>> gW(L), gG(L), gB(L);
  for (int l = 0; l < L; l++){
    gW[l].resize(M.W[l].size());
    gG[l].resize(M.gamma[l].size());
    gB[l].resize(M.beta[l].size());
  }

  auto geri = [&](const int* idx, int m, float lr){
    int nin = M.boyut[L];
    dUst.assign((size_t)m * K, 0.f);
    for (int s = 0; s < m; s++)
      for (int c = 0; c < K; c++)
        dUst[(size_t)s * K + c] = (P[(size_t)s * K + c] - (y[idx[s]] == c ? 1.f : 0.f)) / m;
    const float* Ason = A[L - 1].data();
    std::fill(gWs.begin(), gWs.end(), 0.f);
    std::fill(gBs.begin(), gBs.end(), 0.f);
    #pragma omp parallel for schedule(static)
    for (int c = 0; c < K; c++){
      float* gw = &gWs[(size_t)c * nin];
      float gb = 0;
      for (int s = 0; s < m; s++){
        float dv = dUst[(size_t)s * K + c];
        gb += dv;
        const float* a = Ason + (size_t)s * nin;
        for (int j = 0; j < nin; j++) gw[j] += dv * a[j];
      }
      gBs[c] = gb;
    }
    for (size_t i = 0; i < M.Ws.size(); i++) gWs[i] += l2 * M.Ws[i];
    // dA(L-1)
    dAlt.assign((size_t)m * nin, 0.f);
    #pragma omp parallel for schedule(static)
    for (int s = 0; s < m; s++){
      float* dst = &dAlt[(size_t)s * nin];
      for (int c = 0; c < K; c++){
        float dv = dUst[(size_t)s * K + c];
        const float* w = &M.Ws[(size_t)c * nin];
        for (int j = 0; j < nin; j++) dst[j] += dv * w[j];
      }
    }
    adimSay++;
    adam(M.Ws, gWs, mWs, vWs, lr);
    adam(M.bs, gBs, mBs, vBs, lr);
    std::vector<float> dA = dAlt;
    for (int l = L - 1; l >= 0; l--){
      int nout = M.boyut[l + 1];
      int ninL = M.boyut[l];
      float* Zl = Z[l].data();
      float* ZNl = ZN[l].data();
      uint8_t* Ml = MSK[l].data();
      float tut = 1.f - dropout;
      // dropout + ReLU türevi
      for (int s = 0; s < m; s++){
        size_t t0 = (size_t)s * nout;
        for (int o = 0; o < nout; o++){
          float dv = dA[t0 + o];
          if (!Ml[t0 + o]) dv = 0; else dv /= tut;
          float bnC = M.gamma[l][o] * ZNl[t0 + o] + M.beta[l][o];
          if (bnC <= 0) dv = 0;
          dA[t0 + o] = dv;
        }
      }
      // BN geri yayılımı
      #pragma omp parallel for schedule(static)
      for (int o = 0; o < nout; o++){
        float mu = 0;
        for (int s = 0; s < m; s++) mu += Zl[(size_t)s * nout + o];
        mu /= m;
        float va = 0;
        for (int s = 0; s < m; s++){ float f = Zl[(size_t)s * nout + o] - mu; va += f * f; }
        va /= m;
        float inv = 1.f / std::sqrt(va + BNE);
        float dgn = 0, dbt = 0, dznT = 0, dznZnT = 0;
        for (int s = 0; s < m; s++){
          size_t t = (size_t)s * nout + o;
          float dOut = dA[t];
          dgn += dOut * ZNl[t];
          dbt += dOut;
          float dzn = dOut * M.gamma[l][o];
          dznT += dzn;
          dznZnT += dzn * ZNl[t];
        }
        gG[l][o] = dgn; gB[l][o] = dbt;
        for (int s = 0; s < m; s++){
          size_t t = (size_t)s * nout + o;
          float dzn = dA[t] * M.gamma[l][o];
          dA[t] = inv * (dzn - dznT / m - ZNl[t] * dznZnT / m);
        }
      }
      const float* girdiA = l == 0 ? A0.data() : A[l - 1].data();
      std::fill(gW[l].begin(), gW[l].end(), 0.f);
      #pragma omp parallel for schedule(static)
      for (int o = 0; o < nout; o++){
        float* gw = &gW[l][(size_t)o * ninL];
        for (int s = 0; s < m; s++){
          float dv = dA[(size_t)s * nout + o];
          if (dv == 0) continue;
          const float* a = girdiA + (size_t)s * ninL;
          for (int j = 0; j < ninL; j++) gw[j] += dv * a[j];
        }
      }
      for (size_t i = 0; i < M.W[l].size(); i++) gW[l][i] += l2 * M.W[l][i];
      if (l > 0){
        dAlt.assign((size_t)m * ninL, 0.f);
        #pragma omp parallel for schedule(static)
        for (int s = 0; s < m; s++){
          float* dst = &dAlt[(size_t)s * ninL];
          for (int o = 0; o < nout; o++){
            float dv = dA[(size_t)s * nout + o];
            if (dv == 0) continue;
            const float* w = &M.W[l][(size_t)o * ninL];
            for (int j = 0; j < ninL; j++) dst[j] += dv * w[j];
          }
        }
      }
      adam(M.W[l], gW[l], mW[l], vW[l], lr);
      adam(M.gamma[l], gG[l], mG[l], vG[l], lr);
      adam(M.beta[l], gB[l], mB[l], vB[l], lr);
      if (l > 0) dA = dAlt;
    }
  };

  auto valOlc = [&](std::vector<long long>& Mkar) -> std::pair<double,double> {
    Mkar.assign((size_t)K * K, 0);
    double kayip = 0;
    for (size_t b0 = 0; b0 < val.size(); b0 += parti){
      int m = (int)std::min(val.size() - b0, (size_t)parti);
      ileri(&val[b0], m, false);
      for (int s = 0; s < m; s++){
        int gercek = y[val[b0 + s]];
        int enC = 0;
        for (int c = 1; c < K; c++)
          if (P[(size_t)s * K + c] > P[(size_t)s * K + enC]) enC = c;
        Mkar[(size_t)gercek * K + enC]++;
        kayip -= std::log(std::max(1e-12f, P[(size_t)s * K + gercek]));
      }
    }
    return { val.empty() ? 0 : kayip / val.size(), makroF1(Mkar, K) };
  };

  // en iyi ağırlık anlık görüntüsü
  struct Kopya {
    std::vector<std::vector<float>> W, gamma, beta, runM, runV;
    std::vector<float> Ws, bs;
  } enIyi;
  auto kopyala = [&](){
    enIyi.W = M.W; enIyi.gamma = M.gamma; enIyi.beta = M.beta;
    enIyi.runM = M.runM; enIyi.runV = M.runV; enIyi.Ws = M.Ws; enIyi.bs = M.bs;
  };

  std::vector<int> sira = egit;
  std::vector<long long> Mkar;
  double enIyiF1 = -1;
  int sabirSay = 0;
  bool ilkGecmis = true;
  for (int e = 0; e < enCokDevir; e++){
    float lr = hiz * 0.5f * (1.f + std::cos(3.14159265f * e / enCokDevir));
    G.lr = lr;
    std::shuffle(sira.begin(), sira.end(), rng);
    double kayipTop = 0; long long kayipN = 0;
    for (size_t b0 = 0; b0 < sira.size(); b0 += parti){
      int m = (int)std::min(sira.size() - b0, (size_t)parti);
      ileri(&sira[b0], m, true);
      for (int s = 0; s < m; s++){
        kayipTop -= std::log(std::max(1e-12f, P[(size_t)s * K + y[sira[b0 + s]]]));
        kayipN++;
      }
      geri(&sira[b0], m, lr);
      if (G.durdurIstek) break;
    }
    auto [vKayip, vF1] = valOlc(Mkar);
    G.epoch = e + 1;
    G.kayip = kayipN ? kayipTop / kayipN : 0;
    G.valKayip = vKayip;
    G.valF1 = vF1;
    {
      std::lock_guard<std::mutex> kilit(G.kilit);
      char sat[160];
      snprintf(sat, sizeof(sat), "%s{\"epoch\":%d,\"kayip\":%.5f,\"valKayip\":%.5f,\"valF1\":%.5f,\"lr\":%.6f}",
               ilkGecmis ? "" : ",", e + 1, (double)G.kayip, vKayip, vF1, (double)lr);
      G.gecmisJson += sat;
      ilkGecmis = false;
    }
    if (vF1 > enIyiF1 + 1e-4){
      enIyiF1 = vF1; G.enIyiF1 = vF1; G.enIyiEpoch = e + 1;
      kopyala();
      sabirSay = 0;
    } else if (++sabirSay >= sabir) break;
    if (G.durdurIstek) break;
  }
  if (enIyiF1 >= 0){
    std::lock_guard<std::mutex> kilit(G.kilit);
    M.W = enIyi.W; M.gamma = enIyi.gamma; M.beta = enIyi.beta;
    M.runM = enIyi.runM; M.runV = enIyi.runV; M.Ws = enIyi.Ws; M.bs = enIyi.bs;
  }
  {
    std::lock_guard<std::mutex> kilit(G.kilit);
    G.gecmisJson += "]";
    modelDosyayaKaydet();                 // öğrenilen model bilgisayara kalıcı yazılır
  }
  G.modelHazir = true;
  G.modelDosyadan = false;
  G.egitimde = false;
}

// ============================ HTTP sunucu =============================
static std::string CORS =
  "Access-Control-Allow-Origin: *\r\n"
  "Access-Control-Allow-Methods: GET, POST, OPTIONS\r\n"
  "Access-Control-Allow-Headers: Content-Type\r\n";

static void yanit(soket_t s, int kod, const std::string& tip, const std::string& govde){
  char bas[512];
  const char* durum = kod == 200 ? "200 OK" : (kod == 400 ? "400 Bad Request" : "404 Not Found");
  int n = snprintf(bas, sizeof(bas),
    "HTTP/1.1 %s\r\n%sContent-Type: %s\r\nContent-Length: %zu\r\nConnection: close\r\n\r\n",
    durum, CORS.c_str(), tip.c_str(), govde.size());
  send(s, bas, n, 0);
  size_t g = 0;
  while (g < govde.size()){
    int y = send(s, govde.data() + g, (int)std::min(govde.size() - g, (size_t)65536), 0);
    if (y <= 0) break;
    g += y;
  }
}

static std::string durumJson(){
  std::lock_guard<std::mutex> kilit(G.kilit);
  std::ostringstream o;
  o << "{\"surum\":\"1.0\",\"egitimde\":" << (G.egitimde ? "true" : "false")
    << ",\"modelHazir\":" << (G.modelHazir ? "true" : "false")
    << ",\"epoch\":" << G.epoch << ",\"enCokDevir\":" << G.enCokDevir
    << ",\"kayip\":" << G.kayip << ",\"valKayip\":" << G.valKayip
    << ",\"valF1\":" << G.valF1 << ",\"enIyiF1\":" << G.enIyiF1
    << ",\"enIyiEpoch\":" << G.enIyiEpoch << ",\"lr\":" << G.lr
    << ",\"parametre\":" << G.parametre
    << ",\"modelDosyadan\":" << (G.modelDosyadan ? "true" : "false")
    << ",\"nEgitim\":" << G.nE << ",\"nDogrulama\":" << G.nV
    << ",\"havuz\":{\"n\":" << G.havuz.y.size() << ",\"d\":" << G.havuz.d << ",\"siniflar\":[";
  {
    auto sayim = G.havuz.sayimlar();
    for (size_t i = 0; i < G.havuz.siniflar.size(); i++)
      o << (i ? "," : "") << "{\"ad\":\"" << jsonKacis(G.havuz.siniflar[i])
        << "\",\"n\":" << sayim[i] << "}";
  }
  o << "]}"
    << ",\"boyut\":[";
  for (size_t i = 0; i < G.model.boyut.size(); i++)
    o << (i ? "," : "") << G.model.boyut[i];
  o << "],\"siniflar\":[";
  for (size_t i = 0; i < G.siniflar.size(); i++)
    o << (i ? "," : "") << '"' << jsonKacis(G.siniflar[i]) << '"';
  o << "],\"hata\":\"" << jsonKacis(G.sonHata) << "\"}";
  return o.str();
}

static void istemciIsle(soket_t s){
  std::string veri;
  veri.reserve(65536);
  char tampon[65536];
  size_t govdeBasi = std::string::npos, beklenen = 0;
  for (;;){
    int n = recv(s, tampon, sizeof(tampon), 0);
    if (n <= 0) break;
    veri.append(tampon, n);
    if (govdeBasi == std::string::npos){
      size_t p = veri.find("\r\n\r\n");
      if (p != std::string::npos){
        govdeBasi = p + 4;
        size_t cl = veri.find("Content-Length:");
        if (cl == std::string::npos) cl = veri.find("content-length:");
        beklenen = cl != std::string::npos ? strtoull(veri.c_str() + cl + 15, nullptr, 10) : 0;
      }
    }
    if (govdeBasi != std::string::npos && veri.size() >= govdeBasi + beklenen) break;
    if (veri.size() > (size_t)1 << 31) break;          // 2 GB emniyeti
  }
  if (govdeBasi == std::string::npos){ SOKET_KAPAT(s); return; }
  std::string ilkSatir = veri.substr(0, veri.find("\r\n"));
  std::string govde = veri.substr(govdeBasi);
  auto basliyorMu = [&](const char* on){ return ilkSatir.rfind(on, 0) == 0; };

  if (basliyorMu("OPTIONS")){
    yanit(s, 200, "text/plain", "");
  } else if (basliyorMu("GET /durum")){
    yanit(s, 200, "application/json", durumJson());
  } else if (basliyorMu("POST /durdur")){
    G.durdurIstek = true;
    yanit(s, 200, "application/json", "{\"tamam\":true}");
  } else if (basliyorMu("POST /egit")){
    if (G.egitimde){
      yanit(s, 400, "application/json", "{\"hata\":\"Eğitim zaten sürüyor — önce /durdur çağırın.\"}");
    } else {
      JsonAyristirici ja(govde);
      Json j = ja.coz();
      const Json* jX = j.al("X");
      const Json* jy = j.al("y");
      const Json* jb = j.al("sadeceBu");
      bool sadeceBu = jb && jb->tip == Json::BOOL && jb->dogru;
      std::string hata;
      std::vector<float> X;
      std::vector<int> y;
      std::vector<std::string> adlar;
      int d = 0, K = 0, eklenen = 0;
      {
        std::lock_guard<std::mutex> kilit(G.kilit);
        if (jX && jy && jX->tip == Json::DIZI && jy->tip == Json::DIZI && !jy->dizi.empty()){
          int dGelen = (int)j.sayiAl("d", 36);
          int KGelen = (int)j.sayiAl("K", 0);
          if (ja.hata || KGelen < 2 || dGelen < 1 ||
              jX->dizi.size() != jy->dizi.size() * (size_t)dGelen){
            hata = "Geçersiz istek: d, K, X (n*d) ve y (n) tutarlı olmalı.";
          } else {
            int n = (int)jy->dizi.size();
            std::vector<float> Xg((size_t)n * dGelen);
            std::vector<int> yg(n);
            for (size_t i = 0; i < jX->dizi.size(); i++) Xg[i] = (float)jX->dizi[i].sayi;
            for (int i = 0; i < n; i++){
              yg[i] = (int)jy->dizi[i].sayi;
              if (yg[i] < 0 || yg[i] >= KGelen) hata = "y değerleri 0..K-1 aralığında olmalı.";
            }
            std::vector<std::string> adGelen;
            const Json* js = j.al("siniflar");
            if (js && js->tip == Json::DIZI)
              for (auto& sj : js->dizi) adGelen.push_back(sj.dizgi);
            while ((int)adGelen.size() < KGelen)
              adGelen.push_back("Sınıf " + std::to_string(adGelen.size() + 1));
            if (hata.empty()){
              if (sadeceBu){
                X = std::move(Xg); y = std::move(yg); adlar = adGelen;
                d = dGelen; K = KGelen; eklenen = n;
              } else {
                eklenen = havuzaEkle(dGelen, adGelen, Xg, yg);   // bilgisayara kalıcı yazılır
                if (eklenen < 0)
                  hata = "Öznitelik sayısı havuzla uyuşmuyor (havuz d=" +
                         std::to_string(G.havuz.d) + ", gelen d=" + std::to_string(dGelen) + ").";
              }
            }
          }
        }
        if (hata.empty() && !sadeceBu){
          // birikimli havuzdan eğit (yeni veri geldiyse az önce eklendi)
          if (G.havuz.y.empty()) hata = "Havuz boş — önce eğitim verisi gönderin.";
          else if (G.havuz.siniflar.size() < 2) hata = "Havuzda en az 2 sınıf olmalı.";
          else {
            X = G.havuz.X; y = G.havuz.y; adlar = G.havuz.siniflar;
            d = G.havuz.d; K = (int)adlar.size();
          }
        }
        if (hata.empty()){
          G.siniflar = adlar;
          G.sonHata.clear();
        }
      }
      if (!hata.empty()){
        yanit(s, 400, "application/json", "{\"hata\":\"" + jsonKacis(hata) + "\"}");
      } else {
        const Json* ja2 = j.al("ayarlar");
        long long hedef = ja2 ? (long long)ja2->sayiAl("hedefParam", 1e7) : 10000000LL;
        int devir = ja2 ? (int)ja2->sayiAl("enCokDevir", 120) : 120;
        float hiz = ja2 ? (float)ja2->sayiAl("hiz", 1e-3) : 1e-3f;
        float l2 = ja2 ? (float)ja2->sayiAl("l2", 1e-4) : 1e-4f;
        int parti = ja2 ? (int)ja2->sayiAl("parti", 128) : 128;
        float dropout = ja2 ? (float)ja2->sayiAl("dropout", 0.2) : 0.2f;
        int sabir = ja2 ? (int)ja2->sayiAl("sabir", 12) : 12;
        int n = (int)y.size();
        G.durdurIstek = false;
        G.modelHazir = false;
        G.epoch = 0; G.valF1 = 0; G.enIyiF1 = 0; G.enIyiEpoch = 0;
        G.enCokDevir = devir;
        G.egitimde = true;
        std::thread(egitimCalistir, std::move(X), std::move(y), n, d, K,
                    hedef, devir, hiz, l2, parti, dropout, sabir).detach();
        yanit(s, 200, "application/json",
          "{\"tamam\":true,\"n\":" + std::to_string(n) +
          ",\"eklenen\":" + std::to_string(eklenen) +
          ",\"havuzdan\":" + (sadeceBu ? "false" : "true") + "}");
      }
    }
  } else if (basliyorMu("GET /havuz")){
    std::lock_guard<std::mutex> kilit(G.kilit);
    auto sayim = G.havuz.sayimlar();
    std::ostringstream o;
    o << "{\"n\":" << G.havuz.y.size() << ",\"d\":" << G.havuz.d << ",\"siniflar\":[";
    for (size_t i = 0; i < G.havuz.siniflar.size(); i++)
      o << (i ? "," : "") << "{\"ad\":\"" << jsonKacis(G.havuz.siniflar[i])
        << "\",\"n\":" << sayim[i] << "}";
    o << "]}";
    yanit(s, 200, "application/json", o.str());
  } else if (basliyorMu("POST /havuz/bosalt")){
    std::lock_guard<std::mutex> kilit(G.kilit);
    G.havuz = Havuz{};
    havuzKaydet();
    yanit(s, 200, "application/json", "{\"tamam\":true}");
  } else if (basliyorMu("POST /tahmin")){
    if (!G.modelHazir){
      yanit(s, 400, "application/json", "{\"hata\":\"Önce model eğitin ya da yükleyin.\"}");
    } else {
      JsonAyristirici ja(govde);
      Json j = ja.coz();
      const Json* jX = j.al("X");
      std::lock_guard<std::mutex> kilit(G.kilit);
      int d = G.model.d, K = G.model.K;
      if (ja.hata || !jX || jX->tip != Json::DIZI || jX->dizi.size() % d != 0){
        yanit(s, 400, "application/json",
          "{\"hata\":\"X uzunluğu d=" + std::to_string(d) + " katı olmalı.\"}");
      } else {
        int m = (int)(jX->dizi.size() / d);
        std::vector<float> x(d), probs(K);
        std::ostringstream o;
        o << "{\"probs\":[";
        std::string sinifDizi = ",\"sinif\":[";
        for (int i = 0; i < m; i++){
          for (int jj = 0; jj < d; jj++) x[jj] = (float)jX->dizi[(size_t)i * d + jj].sayi;
          G.model.tahmin(x.data(), probs.data());
          o << (i ? ",[" : "[");
          int enC = 0;
          for (int c = 0; c < K; c++){
            o << (c ? "," : "") << probs[c];
            if (probs[c] > probs[enC]) enC = c;
          }
          o << "]";
          sinifDizi += (i ? "," : "") + std::to_string(enC);
        }
        o << "]" << sinifDizi << "],\"sinifAdlari\":[";
        for (size_t i = 0; i < G.siniflar.size(); i++)
          o << (i ? "," : "") << '"' << jsonKacis(G.siniflar[i]) << '"';
        o << "]}";
        yanit(s, 200, "application/json", o.str());
      }
    }
  } else if (basliyorMu("GET /model")){
    std::lock_guard<std::mutex> kilit(G.kilit);
    std::vector<uint8_t> ikili;
    if (!G.modelHazir || !G.model.kaydet(ikili, G.siniflar)){
      yanit(s, 400, "application/json", "{\"hata\":\"Kaydedilecek model yok.\"}");
    } else {
      std::string govdeB((char*)ikili.data(), ikili.size());
      char bas[512];
      int n = snprintf(bas, sizeof(bas),
        "HTTP/1.1 200 OK\r\n%sContent-Type: application/octet-stream\r\n"
        "Content-Disposition: attachment; filename=\"cam_ai_10m.bin\"\r\n"
        "Content-Length: %zu\r\nConnection: close\r\n\r\n", CORS.c_str(), govdeB.size());
      send(s, bas, n, 0);
      size_t g = 0;
      while (g < govdeB.size()){
        int yz = send(s, govdeB.data() + g, (int)std::min(govdeB.size() - g, (size_t)65536), 0);
        if (yz <= 0) break;
        g += yz;
      }
    }
  } else if (basliyorMu("POST /model")){
    std::lock_guard<std::mutex> kilit(G.kilit);
    if (G.model.yukle((const uint8_t*)govde.data(), govde.size(), &G.siniflar)){
      G.modelHazir = true;
      G.modelDosyadan = true;
      G.parametre = G.model.parametre;
      modelDosyayaKaydet();               // yüklenen model de kalıcı olsun
      yanit(s, 200, "application/json",
        "{\"tamam\":true,\"parametre\":" + std::to_string(G.model.parametre) + "}");
    } else {
      yanit(s, 400, "application/json", "{\"hata\":\"Geçersiz model dosyası (CAMAI10M imzası bekleniyor).\"}");
    }
  } else {
    yanit(s, 404, "application/json", "{\"hata\":\"Bilinmeyen uç: /durum /egit /durdur /tahmin /model\"}");
  }
  SOKET_KAPAT(s);
}

int main(int argc, char** argv){
  int port = argc > 1 ? atoi(argv[1]) : 8787;
#ifdef _WIN32
  WSADATA wsa;
  WSAStartup(MAKEWORD(2, 2), &wsa);
#endif
  soket_t dinleyici = socket(AF_INET, SOCK_STREAM, 0);
  int bir = 1;
  setsockopt(dinleyici, SOL_SOCKET, SO_REUSEADDR, (const char*)&bir, sizeof(bir));
  sockaddr_in adres{};
  adres.sin_family = AF_INET;
  adres.sin_port = htons((uint16_t)port);
  adres.sin_addr.s_addr = htonl(INADDR_LOOPBACK);      // yalnız bu bilgisayardan erişim
  if (bind(dinleyici, (sockaddr*)&adres, sizeof(adres)) != 0){
    fprintf(stderr, "HATA: %d portu kullanılamıyor (başka kopya açık olabilir).\n", port);
    return 1;
  }
  listen(dinleyici, 16);
  /* önceki oturumda öğrenilenleri diskten geri yükle */
  {
    std::lock_guard<std::mutex> kilit(G.kilit);
    havuzYukle();
    std::ifstream f(MODEL_DOSYA, std::ios::binary);
    if (f){
      std::vector<uint8_t> ikili((std::istreambuf_iterator<char>(f)), std::istreambuf_iterator<char>());
      if (G.model.yukle(ikili.data(), ikili.size(), &G.siniflar)){
        G.modelHazir = true;
        G.modelDosyadan = true;
        G.parametre = G.model.parametre;
      }
    }
  }
  printf("🌲 Çam-AI yerel eğitim sunucusu hazır: http://localhost:%d\n", port);
  if (G.modelHazir)
    printf("   Kayıtlı model yüklendi (%s, %lld parametre).\n", MODEL_DOSYA, (long long)G.parametre);
  if (!G.havuz.y.empty())
    printf("   Kayıtlı veri havuzu: %zu örnek, %zu sınıf (%s).\n",
           G.havuz.y.size(), G.havuz.siniflar.size(), HAVUZ_DOSYA);
  printf("   kopru.html dosyasını tarayıcıda açıp bağlanın. Kapatmak: Ctrl+C\n");
  fflush(stdout);
  for (;;){
    soket_t s = accept(dinleyici, nullptr, nullptr);
    if (s == INVALID_SOCKET) continue;
    std::thread(istemciIsle, s).detach();
  }
  return 0;
}
