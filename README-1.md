# BIST Backtest – İdeal İndikatör Değerleri

Bu proje, BIST hisseleri için teknik filtrelerin en uygun değerlerini bulmaya yönelik bir backtest altyapısıdır.

## Ne yapar?

- BIST hisse listesini `bist_tum.txt` dosyasından okur
- Her hisse için günlük veriyi `yfinance` ile indirir
- Şu filtreleri test eder:
  - MA fast > MA slow
  - Fiyat > MA fast
  - RSI belirli aralıkta
  - Relative Volume eşiği
  - ADX eşiği
  - Son N gün direnci kırılmış
- Walk-forward (ileri testli) optimizasyon yapar
- OOS (out-of-sample) sonuçları CSV olarak kaydeder

## Kurulum

```bash
pip install -r requirements.txt
```

## Çalıştırma

Hızlı ilk tarama:

```bash
python backtest.py --symbols-file bist_tum.txt --quick
```

Tam tarama:

```bash
python backtest.py --symbols-file bist_tum.txt
```

## Çıktılar

`backtest_output/` klasörüne şunlar yazılır:

- `symbol_status.csv`
- `training_parameter_summary.csv`
- `walk_forward_folds.csv`
- `oos_trades.csv`
- `best_parameter_sets.csv`
- `run_summary.json`

## Not

- Bu sistem yatırım tavsiyesi değildir.
- En iyi sonuçlar sadece toplam kâr ile değil; trade sayısı, win rate, profit factor ve maksimum kayıp ile birlikte değerlendirilmelidir.
- BIST’te düşük hacimli hisselerde veri kalitesi değişken olabilir.
