# 🤖 BIST AI Al-Sat Robotu — Kurulum Rehberi

## 📋 Gereksinimler

- Python 3.9+
- İnternet bağlantısı
- (Opsiyonel) Telegram Bot Token
- (Opsiyonel) TradingView Premium (Webhook için)

-----

## 🚀 Kurulum

### 1. Kütüphaneleri Yükle

```bash
pip install -r requirements.txt
```

### 2. Ayarları Düzenle

`trading_bot.py` dosyasını aç ve `CONFIG` bölümünü düzenle:

```python
CONFIG = {
    # Takip etmek istediğin BIST hisseleri
    "hisseler": ["THYAO.IS", "GARAN.IS", "ASELS.IS"],

    # TradingView Pro+ webhook URL'si (opsiyonel)
    "tradingview_webhook": "https://...",

    # Telegram bildirimleri (opsiyonel)
    "telegram_token": "BOT_TOKEN_BURAYA",
    "telegram_chat_id": "CHAT_ID_BURAYA",

    # Her kaç saniyede tarama yapılsın (3600 = 1 saat)
    "tarama_suresi": 3600,
}
```

### 3. Botu Çalıştır

```bash
python trading_bot.py
```

-----

## 📡 API Endpoint’leri

|Endpoint  |Metot|Açıklama                 |
|----------|-----|-------------------------|
|`/webhook`|POST |TradingView’dan sinyal al|
|`/tarama` |GET  |Manuel tarama başlat     |
|`/durum`  |GET  |Bot durumunu kontrol et  |

### Manuel Tarama Örneği:

```bash
curl http://localhost:5000/tarama
```

-----

## 📊 Nasıl Çalışır?

1. **Veri Çekme**: Yahoo Finance’dan BIST hisse verileri indirilir
1. **Göstergeler**: RSI, MACD, Bollinger Bands, Stochastic hesaplanır
1. **AI Modeli**: Random Forest ile 5 günlük fiyat yönü tahmin edilir
1. **Sinyal**: AL 🟢 / SAT 🔴 sinyali üretilir
1. **Bildirim**: Telegram ve/veya TradingView webhook’a gönderilir

-----

## ⚠️ Önemli Uyarılar

> **Bu bot yatırım tavsiyesi değildir!**
> Finansal kararlar almadan önce mutlaka bir uzmanla görüşün.
> Geçmiş performans gelecekteki sonuçları garanti etmez.

-----

## 🔧 TradingView Entegrasyonu

1. TradingView’da bir **Alert** oluştur
1. “Webhook URL” kısmına bot adresini gir:
   
   ```
   http://SUNUCU_IP:5000/webhook
   ```
1. JSON mesaj formatı:
   
   ```json
   {"ticker": "{{ticker}}", "action": "buy", "price": {{close}}}
   ```

-----

## 📱 Telegram Bot Kurulumu

1. Telegram’da `@BotFather`’a mesaj at
1. `/newbot` komutuyla yeni bot oluştur
1. Aldığın token’ı `CONFIG["telegram_token"]`’a yapıştır
1. `@userinfobot`‘tan Chat ID’ni öğren
1. `CONFIG["telegram_chat_id"]`’ye yapıştır