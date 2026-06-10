"""
BIST Yapay Zeka Al-Sat Robotu
==============================
- Yahoo Finance'dan BIST verisi çeker
- Random Forest + teknik göstergelerle sinyal üretir
- TradingView Webhook sinyali gönderir
- Flask sunucusu ile webhook alır
"""

import numpy as np
import os
import pandas as pd
import yfinance as yf
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import requests
import json
import time
import logging
from datetime import datetime, timedelta
from flask import Flask, request, jsonify
import threading
import warnings
warnings.filterwarnings('ignore')

# ─────────────────────────────────────────────
# AYARLAR
# ─────────────────────────────────────────────
CONFIG = {
    "hisseler": [
        # BIST 100 Hisseleri
        "THYAO.IS", "GARAN.IS", "ASELS.IS", "SISE.IS", "AKBNK.IS",
        "EREGL.IS", "KCHOL.IS", "BIMAS.IS", "TOASO.IS", "FROTO.IS",
        "TCELL.IS", "ENKAI.IS", "SAHOL.IS", "YKBNK.IS", "HALKB.IS",
        "VAKBN.IS", "PGSUS.IS", "KOZAL.IS", "KRDMD.IS", "PETKM.IS",
        "TUPRS.IS", "ARCLK.IS", "VESTL.IS", "EKGYO.IS", "EMLAK.IS",
        "TAVHL.IS", "LOGO.IS", "NETAS.IS", "TTKOM.IS", "DOHOL.IS",
        "SOKM.IS", "MGROS.IS", "ULKER.IS", "CCOLA.IS", "AEFES.IS",
        "AGHOL.IS", "AKSEN.IS", "ALARK.IS", "ALGYO.IS", "ALKIM.IS",
        "ASUZU.IS", "ATAKP.IS", "AVISA.IS", "AYDEM.IS", "AYGAZ.IS",
        "BAGFS.IS", "BANVT.IS", "BIOEN.IS", "BRISA.IS", "BRYAT.IS",
        "BSOKE.IS", "BTCIM.IS", "CANTE.IS", "CEMTS.IS", "CIMSA.IS",
        "CLEBI.IS", "CMENT.IS", "COTAS.IS", "CUSAN.IS", "DESA.IS",
        "DEVA.IS", "DGKLB.IS", "DOAS.IS", "DOBUR.IS", "DURDO.IS",
        "DYOBY.IS", "ECZYT.IS", "EGEEN.IS", "EGPRO.IS", "ENERY.IS",
        "ERBOS.IS", "EREGL.IS", "ERSU.IS", "ESCOM.IS", "EUPWR.IS",
        "FENER.IS", "FLAP.IS", "FMIZP.IS", "FORMT.IS", "GLYHO.IS",
        "GMTAS.IS", "GOODY.IS", "GOZDE.IS", "GUBRF.IS", "GWIND.IS",
        "HEKTS.IS", "HLGYO.IS", "HRKET.IS", "ICBCT.IS", "IEYHO.IS",
        "IHLGM.IS", "IHLAS.IS", "INDES.IS", "IPEKE.IS", "ISATR.IS",
        "ISCTR.IS", "ISGYO.IS", "ISKUR.IS", "ISMEN.IS", "IZMDC.IS",
    ],
    "periyot": "1y",           # Geçmiş veri süresi
    "interval": "1d",          # Günlük mum
    "tradingview_webhook": "",  # TradingView webhook URL'nizi buraya girin
    "telegram_token": os.environ.get("TELEGRAM_TOKEN", ""),       # Telegram bot token
    "telegram_chat_id": os.environ.get("TELEGRAM_CHAT_ID", ""),     # Telegram chat ID
    "flask_port": 5000,
    "tarama_suresi": 3600,      # Her kaç saniyede tarama (3600 = 1 saat)
}

# ─────────────────────────────────────────────
# ÖNCEKİ SİNYALLER (Dönüş tespiti için)
# ─────────────────────────────────────────────
onceki_sinyaller = {}  # {ticker: "AL" veya "SAT"}

# ─────────────────────────────────────────────
# LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler("bot.log"),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# ─────────────────────────────────────────────
# TEKNİK GÖSTERGELER
# ─────────────────────────────────────────────
def teknik_gostergeler_hesapla(df: pd.DataFrame) -> pd.DataFrame:
    """Tüm teknik göstergeleri hesaplar."""

    # RSI
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(14).mean()
    avg_loss = loss.rolling(14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))

    # MACD
    ema12 = df['Close'].ewm(span=12).mean()
    ema26 = df['Close'].ewm(span=26).mean()
    df['MACD'] = ema12 - ema26
    df['MACD_Signal'] = df['MACD'].ewm(span=9).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']

    # Bollinger Bands
    ma20 = df['Close'].rolling(20).mean()
    std20 = df['Close'].rolling(20).std()
    df['BB_Upper'] = ma20 + 2 * std20
    df['BB_Lower'] = ma20 - 2 * std20
    bb_range = (df['BB_Upper'] - df['BB_Lower']).replace(0, float('nan'))
    df['BB_Width'] = bb_range / ma20.replace(0, float('nan'))
    df['BB_Width'] = df['BB_Width'].fillna(0)
    df['BB_Position'] = (df['Close'] - df['BB_Lower']) / bb_range
    df['BB_Position'] = df['BB_Position'].fillna(0.5)

    # Hareketli Ortalamalar
    df['MA5']  = df['Close'].rolling(5).mean()
    df['MA10'] = df['Close'].rolling(10).mean()
    df['MA20'] = df['Close'].rolling(20).mean()
    df['MA50'] = df['Close'].rolling(50).mean()

    # MA Crossover sinyalleri
    df['MA5_10_Cross']  = (df['MA5'] > df['MA10']).astype(int)
    df['MA10_20_Cross'] = (df['MA10'] > df['MA20']).astype(int)

    # Volume göstergeleri
    df['Volume_MA'] = df['Volume'].rolling(20).mean()
    df['Volume_Ratio'] = df['Volume'] / df['Volume_MA']

    # Stochastic Oscillator
    low14  = df['Low'].rolling(14).min()
    high14 = df['High'].rolling(14).max()
    df['Stoch_K'] = 100 * (df['Close'] - low14) / (high14 - low14)
    df['Stoch_D'] = df['Stoch_K'].rolling(3).mean()

    # ATR (Average True Range)
    tr1 = df['High'] - df['Low']
    tr2 = abs(df['High'] - df['Close'].shift())
    tr3 = abs(df['Low']  - df['Close'].shift())
    df['ATR'] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean()

    # Fiyat değişim oranları
    df['Return_1d'] = df['Close'].pct_change(1)
    df['Return_5d'] = df['Close'].pct_change(5)
    df['Return_10d'] = df['Close'].pct_change(10)

    # Hedef: 5 günlük getiri pozitif mi?
    df['Target'] = (df['Close'].shift(-5) > df['Close']).astype(int)

    return df.dropna()


# ─────────────────────────────────────────────
# MODEL
# ─────────────────────────────────────────────
class BISTModel:
    def __init__(self):
        self.model = RandomForestClassifier(
            n_estimators=200,
            max_depth=10,
            min_samples_split=5,
            random_state=42,
            n_jobs=-1
        )
        self.scaler = StandardScaler()
        self.ozellikler = [
            'RSI', 'MACD', 'MACD_Signal', 'MACD_Hist',
            'BB_Width', 'BB_Position',
            'MA5_10_Cross', 'MA10_20_Cross',
            'Volume_Ratio', 'Stoch_K', 'Stoch_D',
            'ATR', 'Return_1d', 'Return_5d', 'Return_10d'
        ]
        self.egitildi = False

    def egit(self, df: pd.DataFrame) -> dict:
        """Modeli verilen DataFrame ile eğitir."""
        X = df[self.ozellikler]
        y = df['Target']

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, shuffle=False
        )

        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled  = self.scaler.transform(X_test)

        self.model.fit(X_train_scaled, y_train)
        self.egitildi = True

        accuracy = self.model.score(X_test_scaled, y_test)
        log.info(f"Model doğruluğu: {accuracy:.2%}")
        return {"accuracy": accuracy}

    def tahmin_et(self, df: pd.DataFrame) -> dict:
        """Son satır için tahmin üretir."""
        if not self.egitildi:
            raise ValueError("Model henüz eğitilmedi!")

        son_satir = df[self.ozellikler].iloc[-1:]
        son_satir_scaled = self.scaler.transform(son_satir)

        tahmin   = self.model.predict(son_satir_scaled)[0]
        olasilik = self.model.predict_proba(son_satir_scaled)[0]

        sinyal = "AL 🟢" if tahmin == 1 else "SAT 🔴"
        guc    = max(olasilik)

        return {
            "sinyal": sinyal,
            "tahmin": int(tahmin),
            "al_olasiligi":  round(float(olasilik[1]), 4),
            "sat_olasiligi": round(float(olasilik[0]), 4),
            "guc": round(float(guc), 4)
        }


# ─────────────────────────────────────────────
# VERİ & TAHMİN
# ─────────────────────────────────────────────
def hisse_analiz_et(ticker: str) -> dict:
    """Bir hisseyi indir, eğit ve sinyal üret."""
    log.info(f"Analiz ediliyor: {ticker}")

    df = yf.download(ticker, period=CONFIG["periyot"],
                     interval=CONFIG["interval"], progress=False,
                     auto_adjust=True)

    # yfinance yeni versiyonda multi-level columns döndürüyor, düzelt
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    if df.empty or len(df) < 60:
        log.warning(f"{ticker} için yeterli veri yok.")
        return {"hata": "Yetersiz veri"}

    df = teknik_gostergeler_hesapla(df)

    model = BISTModel()
    egitim_sonucu = model.egit(df)
    tahmin = model.tahmin_et(df)

    son_fiyat = float(df['Close'].iloc[-1])
    tarih     = df.index[-1].strftime("%Y-%m-%d")

    return {
        "ticker":      ticker,
        "tarih":       tarih,
        "son_fiyat":   round(son_fiyat, 2),
        "sinyal":      tahmin["sinyal"],
        "al_olasiligi":  tahmin["al_olasiligi"],
        "sat_olasiligi": tahmin["sat_olasiligi"],
        "guc":         tahmin["guc"],
        "model_dogrulugu": round(egitim_sonucu["accuracy"], 4),
    }


# ─────────────────────────────────────────────
# BİLDİRİM
# ─────────────────────────────────────────────
def telegram_gonder(mesaj: str):
    if not CONFIG["telegram_token"] or not CONFIG["telegram_chat_id"]:
        return
    url = f"https://api.telegram.org/bot{CONFIG['telegram_token']}/sendMessage"
    payload = {"chat_id": CONFIG["telegram_chat_id"], "text": mesaj, "parse_mode": "HTML"}
    try:
        requests.post(url, json=payload, timeout=5)
    except Exception as e:
        log.error(f"Telegram hatası: {e}")


def tradingview_alert_gonder(ticker: str, sinyal: str, fiyat: float):
    if not CONFIG["tradingview_webhook"]:
        return
    payload = {
        "ticker":  ticker,
        "action":  "buy" if "AL" in sinyal else "sell",
        "price":   fiyat,
        "time":    datetime.now().isoformat()
    }
    try:
        r = requests.post(CONFIG["tradingview_webhook"],
                          json=payload, timeout=5)
        log.info(f"TradingView webhook gönderildi: {r.status_code}")
    except Exception as e:
        log.error(f"Webhook hatası: {e}")


# ─────────────────────────────────────────────
# ANA TARAMA DÖNGÜSÜ
# ─────────────────────────────────────────────
def tarama_yap():
    """Tüm hisseleri tara ve sinyalleri raporla."""
    global onceki_sinyaller
    log.info("=" * 50)
    log.info(f"Tarama başladı: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    sonuclar = []
    donus_sinyalleri = []  # SAT->AL dönüşleri

    for ticker in CONFIG["hisseler"]:
        try:
            sonuc = hisse_analiz_et(ticker)
            sonuclar.append(sonuc)

            guncel_sinyal = "AL" if "AL" in sonuc['sinyal'] else "SAT"
            onceki_sinyal = onceki_sinyaller.get(ticker, None)

            # Dönüş tespiti: SAT -> AL
            donus = onceki_sinyal == "SAT" and guncel_sinyal == "AL"
            if donus:
                donus_sinyalleri.append(sonuc)

            if donus:
                donus_prefix = "Donus Sinyali!\n"
                donus_suffix = " (Onceki: SAT)"
            else:
                donus_prefix = ""
                donus_suffix = ""
            al_oran = "{:.1%}".format(sonuc['al_olasiligi'])
            model_oran = "{:.1%}".format(sonuc['model_dogrulugu'])
            mesaj = (
                donus_prefix +
                "Hisse: " + sonuc['ticker'] + "\n" +
                "Tarih: " + sonuc['tarih'] + "\n" +
                "Fiyat: " + str(sonuc['son_fiyat']) + " TL\n" +
                "Sinyal: " + sonuc['sinyal'] + donus_suffix + "\n" +
                "Al Olasiligi: " + al_oran + "\n" +
                "Model Dogrulugu: " + model_oran
            )
            log.info(mesaj.replace("<b>", "").replace("</b>", ""))

            # Bildirim: dönüş sinyali VEYA %65+ doğruluk
            if donus and sonuc['model_dogrulugu'] >= 0.65:
                telegram_gonder(mesaj)
            elif sonuc['model_dogrulugu'] >= 0.65:
                telegram_gonder(mesaj)
            else:
                log.info(f"{ticker} doğruluk düşük, bildirim gönderilmedi.")

            # Sinyali kaydet
            onceki_sinyaller[ticker] = guncel_sinyal

            tradingview_alert_gonder(
                sonuc['ticker'], sonuc['sinyal'], sonuc['son_fiyat']
            )
            time.sleep(2)  # API hız limiti
        except Exception as e:
            log.error(f"{ticker} hatası: {e}")

    # Tarama sonu özet: dönüş sinyalleri
    if donus_sinyalleri:
        ozet = "DONUS SINYALLERI (SAT->AL)\n\n"
        for s in donus_sinyalleri:
            if s['model_dogrulugu'] >= 0.65:
                ozet += s["ticker"] + " - " + str(s["son_fiyat"]) + " TL - Al Olasiligi: " + "{:.1%}".format(s["al_olasiligi"]) + "\n"
        telegram_gonder(ozet)

    log.info("Tarama tamamlandı.")
    return sonuclar


def periyodik_tarama():
    """Hafta içi 09:30, 13:30, 17:30 tarama yapar."""
    TARAMA_SAATLERI = [(9, 30), (13, 30), (17, 30)]
    while True:
        simdi = datetime.now()
        # Hafta sonu ise bekle
        if simdi.weekday() >= 5:
            log.info("Hafta sonu, bekleniyor...")
            time.sleep(3600)
            continue
        # Sonraki tarama saatini bul
        sonraki = None
        for saat, dakika in TARAMA_SAATLERI:
            hedef = simdi.replace(hour=saat, minute=dakika, second=0, microsecond=0)
            if hedef > simdi:
                sonraki = hedef
                break
        # Bugün tarama saati kalmadıysa yarın ilk saate ayarla
        if sonraki is None:
            yarin = simdi + timedelta(days=1)
            while yarin.weekday() >= 5:
                yarin += timedelta(days=1)
            sonraki = yarin.replace(hour=9, minute=30, second=0, microsecond=0)
        bekle = (sonraki - simdi).total_seconds()
        log.info(f"Sonraki tarama: {sonraki.strftime('%Y-%m-%d %H:%M')} ({int(bekle/60)} dakika sonra)")
        time.sleep(bekle)
        tarama_yap()


# ─────────────────────────────────────────────
# FLASK API (TradingView Webhook Alıcısı)
# ─────────────────────────────────────────────
app = Flask(__name__)

@app.route("/webhook", methods=["POST"])
def webhook_al():
    """TradingView'dan gelen webhook'u işle."""
    data = request.get_json()
    log.info(f"Webhook alındı: {data}")
    return jsonify({"durum": "ok", "veri": data}), 200

@app.route("/tarama", methods=["GET"])
def manuel_tarama():
    """Manuel tarama tetikle."""
    sonuclar = tarama_yap()
    return jsonify(sonuclar), 200

@app.route("/durum", methods=["GET"])
def durum():
    return jsonify({"durum": "çalışıyor", "zaman": datetime.now().isoformat()}), 200


# ─────────────────────────────────────────────
# BAŞLATMA
# ─────────────────────────────────────────────
if __name__ == "__main__":
    log.info("🤖 BIST AI Al-Sat Robotu Başlatılıyor...")

    # İlk taramayı arka planda başlat
    tarama_thread = threading.Thread(target=periyodik_tarama, daemon=True)
    tarama_thread.start()

    # Flask sunucusunu başlat
    app.run(host="0.0.0.0", port=CONFIG["flask_port"], debug=False)
