# BIST Backtest - Chunked Version

Bu repo, uzun süren backtest işlemlerini Streamlit içinde daha güvenli çalıştırmak için iki parçaya ayrılmıştır.

- `app.py` → yalnızca arayüz
- `backtest.py` → hesap motoru
- `results/` → oluşan CSV çıktıları

## Özellik

- BIST sembol listesi `bist_tum.txt` dosyasından okunur
- İşlem tek seferde tüm hisseleri değil, küçük parçalar halinde yürür
- Her parça sonrası sonuçlar `results/` klasörüne yazılır

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

```bash
streamlit run app.py
```

## Gerekli dosya

Repo kökünde şu dosya bulunmalıdır:

- `bist_tum.txt`

## Not

Bu uygulama yatırım tavsiyesi değildir.
