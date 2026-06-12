import streamlit as st
import pandas as pd
from pathlib import Path

from backtest import (
    read_symbols,
    build_param_grid,
    load_ohlcv,
    process_symbol,
)

st.set_page_config(page_title="BIST Backtest", layout="wide")
st.title("📊 BIST Backtest")
st.caption("İdeal indikatör değerlerini bulmak için walk-forward backtest")

st.sidebar.header("Ayarlar")
symbols_file = st.sidebar.text_input("Sembol dosyası", "bist_tum.txt")
period = st.sidebar.selectbox("Veri süresi", ["5y", "10y", "max"], index=1)
interval = st.sidebar.selectbox("Zaman aralığı", ["1d"], index=0)
quick = st.sidebar.checkbox("Hızlı mod", value=True)
max_symbols = st.sidebar.number_input("Maksimum hisse", min_value=1, value=20, step=1)

run_btn = st.sidebar.button("Backtest çalıştır")

if run_btn:
    symbols_path = Path(symbols_file)
    if not symbols_path.exists():
        st.error(f"Sembol dosyası bulunamadı: {symbols_path.resolve()}")
        st.stop()

    with st.spinner("Semboller okunuyor..."):
        symbols = read_symbols(symbols_path)

    symbols = symbols[: int(max_symbols)]

    if quick:
        param_grid = build_param_grid(
            ma_fast_list=[20, 50],
            ma_slow_list=[100, 200],
            rsi_period_list=[7, 14],
            rsi_low_list=[50, 55],
            rsi_high_list=[65, 70],
            rel_vol_list=[1.2, 1.5],
            adx_list=[18, 20, 25],
            breakout_list=[10, 20],
            hold_days_list=[5, 10],
            stop_loss_list=[0.05],
            take_profit_list=[0.10],
        )
    else:
        param_grid = build_param_grid(
            ma_fast_list=[20, 30, 50],
            ma_slow_list=[100, 150, 200],
            rsi_period_list=[7, 14],
            rsi_low_list=[50, 52, 55, 58],
            rsi_high_list=[65, 68, 70, 75],
            rel_vol_list=[1.2, 1.5, 1.8],
            adx_list=[18, 20, 22, 25],
            breakout_list=[10, 20, 30],
            hold_days_list=[5, 10, 15],
            stop_loss_list=[0.03, 0.05, 0.08],
            take_profit_list=[0.06, 0.10, 0.15],
        )

    st.info(f"{len(symbols)} hisse taranacak. Parametre kombinasyonu: {len(param_grid)}")

    progress = st.progress(0)
    status_box = st.empty()

    all_results = []

    for i, symbol in enumerate(symbols, start=1):
        status_box.write(f"İşleniyor: {symbol} ({i}/{len(symbols)})")

        result = process_symbol(
            symbol=symbol,
            period=period,
            interval=interval,
            cache_dir=Path("cache"),
            param_grid=param_grid,
            train_size=504,
            test_size=126,
            step_size=126,
        )

        all_results.append(
            {
                "symbol": symbol,
                "status": result["status"],
                "best_params": result["train_summary"].iloc[0].to_dict() if not result["train_summary"].empty else None,
            }
        )

        progress.progress(i / len(symbols))

    df_results = pd.DataFrame(all_results)

    st.subheader("Tarama Sonuçları")
    st.dataframe(df_results, use_container_width=True)

    ok_count = (df_results["status"] == "ok").sum()
    st.success(f"Tarama tamamlandı. Uygun veri bulunan hisse sayısı: {ok_count}")
else:
    st.info("Sol menüden ayar yapıp **Backtest çalıştır** butonuna basın.")
