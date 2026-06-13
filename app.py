from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pandas as pd
import streamlit as st

from backtest import (
    build_param_grid,
    process_symbol,
    read_symbols,
    resolve_symbols_path,
    save_results,
)

st.set_page_config(page_title="BIST Backtest", layout="wide")
st.title("📊 BIST Backtest")
st.caption("İdeal indikatör değerlerini bulmak için walk-forward backtest")

st.sidebar.header("Ayarlar")
symbols_file_input = st.sidebar.text_input("Sembol dosyası", "bist_tum.txt")
period = st.sidebar.selectbox("Veri süresi", ["5y", "10y", "max"], index=1)
interval = st.sidebar.selectbox("Zaman aralığı", ["1d"], index=0)
quick = st.sidebar.checkbox("Hızlı mod", value=True)
max_symbols = st.sidebar.number_input("Maksimum hisse", min_value=1, value=20, step=1)
batch_size = st.sidebar.slider("Her çalışmada işlenecek hisse sayısı", min_value=1, max_value=10, value=3)
auto_continue = st.sidebar.checkbox("Bitişe kadar otomatik devam et", value=True)

train_size = st.sidebar.number_input("Train bar sayısı", min_value=100, value=504, step=21)
test_size = st.sidebar.number_input("Test bar sayısı", min_value=20, value=126, step=21)
step_size = st.sidebar.number_input("İleri kayma bar sayısı", min_value=20, value=126, step=21)

results_dir = Path("results")
cache_dir = Path("cache")

def make_param_grid(quick_mode: bool):
    if quick_mode:
        return build_param_grid(
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
    return build_param_grid(
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

if "job" not in st.session_state:
    st.session_state.job = {
        "running": False,
        "done": False,
        "symbols": [],
        "cursor": 0,
        "status_rows": [],
        "train_rows": [],
        "fold_rows": [],
        "trade_rows": [],
        "started_at": None,
        "symbol_file": None,
        "period": None,
        "interval": None,
        "quick": None,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "step_size": int(step_size),
        "param_grid_size": 0,
    }

col_a, col_b = st.columns([1, 1])

with col_a:
    start_clicked = st.button("Backtest başlat / yenile", use_container_width=True)
with col_b:
    reset_clicked = st.button("Sıfırla", use_container_width=True)

if reset_clicked:
    st.session_state.job = {
        "running": False,
        "done": False,
        "symbols": [],
        "cursor": 0,
        "status_rows": [],
        "train_rows": [],
        "fold_rows": [],
        "trade_rows": [],
        "started_at": None,
        "symbol_file": None,
        "period": None,
        "interval": None,
        "quick": None,
        "train_size": int(train_size),
        "test_size": int(test_size),
        "step_size": int(step_size),
        "param_grid_size": 0,
    }
    st.rerun()

job = st.session_state.job

if start_clicked and not job["running"]:
    try:
        symbols_path = resolve_symbols_path(symbols_file_input)
        symbols = read_symbols(symbols_path)
        symbols = symbols[: int(max_symbols)]
        param_grid = make_param_grid(quick)
        job.update(
            {
                "running": True,
                "done": False,
                "symbols": symbols,
                "cursor": 0,
                "status_rows": [],
                "train_rows": [],
                "fold_rows": [],
                "trade_rows": [],
                "started_at": datetime.utcnow().isoformat(timespec="seconds"),
                "symbol_file": str(symbols_path),
                "period": period,
                "interval": interval,
                "quick": bool(quick),
                "train_size": int(train_size),
                "test_size": int(test_size),
                "step_size": int(step_size),
                "param_grid_size": len(param_grid),
            }
        )
        st.success(f"{len(symbols)} hisse yüklendi. Parametre kombinasyonu: {len(param_grid)}")
    except Exception as e:
        st.error(str(e))
        st.stop()

if job["running"] and job["symbols"]:
    symbols = job["symbols"]
    cursor = int(job["cursor"])
    end = min(cursor + int(batch_size), len(symbols))
    batch = symbols[cursor:end]

    st.info(f"Çalışıyor: {cursor}/{len(symbols)} tamamlandı. Bu turda {len(batch)} hisse işlenecek.")

    progress = st.progress(cursor / max(len(symbols), 1))
    status_box = st.empty()
    status_box.write(f"İşlenecek aralık: {cursor + 1}-{end}")

    param_grid = make_param_grid(job["quick"])

    for idx, symbol in enumerate(batch, start=cursor + 1):
        status_box.write(f"İşleniyor: {symbol} ({idx}/{len(symbols)})")
        try:
            result = process_symbol(
                symbol=symbol,
                period=job["period"],
                interval=job["interval"],
                cache_dir=cache_dir,
                param_grid=param_grid,
                train_size=job["train_size"],
                test_size=job["test_size"],
                step_size=job["step_size"],
            )

            job["status_rows"].append(
                {
                    "symbol": symbol,
                    "status": result.get("status", "unknown"),
                    "folds": len(result["fold_summary"]) if isinstance(result.get("fold_summary"), pd.DataFrame) else 0,
                    "train_rows": len(result["train_summary"]) if isinstance(result.get("train_summary"), pd.DataFrame) else 0,
                    "oos_trades": len(result["oos_trades"]) if isinstance(result.get("oos_trades"), pd.DataFrame) else 0,
                }
            )

            if isinstance(result.get("train_summary"), pd.DataFrame) and not result["train_summary"].empty:
                ts = result["train_summary"].copy()
                ts.insert(0, "symbol", symbol)
                job["train_rows"].append(ts)

            if isinstance(result.get("fold_summary"), pd.DataFrame) and not result["fold_summary"].empty:
                fs = result["fold_summary"].copy()
                fs.insert(0, "symbol", symbol)
                job["fold_rows"].append(fs)

            if isinstance(result.get("oos_trades"), pd.DataFrame) and not result["oos_trades"].empty:
                ot = result["oos_trades"].copy()
                ot.insert(0, "symbol", symbol)
                job["trade_rows"].append(ot)

        except Exception as e:
            job["status_rows"].append(
                {
                    "symbol": symbol,
                    "status": f"error: {e}",
                    "folds": 0,
                    "train_rows": 0,
                    "oos_trades": 0,
                }
            )

        job["cursor"] = idx
        progress.progress(job["cursor"] / max(len(symbols), 1))

    status_df = pd.DataFrame(job["status_rows"])
    train_df = pd.concat(job["train_rows"], ignore_index=True) if job["train_rows"] else pd.DataFrame()
    fold_df = pd.concat(job["fold_rows"], ignore_index=True) if job["fold_rows"] else pd.DataFrame()
    trade_df = pd.concat(job["trade_rows"], ignore_index=True) if job["trade_rows"] else pd.DataFrame()

    save_results(results_dir, status_df, train_df, fold_df, trade_df)

    if job["cursor"] >= len(symbols):
        job["running"] = False
        job["done"] = True
    else:
        if auto_continue:
            st.rerun()

if job["done"]:
    st.success("Backtest tamamlandı.")

    status_df = pd.DataFrame(job["status_rows"])
    train_df = pd.concat(job["train_rows"], ignore_index=True) if job["train_rows"] else pd.DataFrame()
    fold_df = pd.concat(job["fold_rows"], ignore_index=True) if job["fold_rows"] else pd.DataFrame()
    trade_df = pd.concat(job["trade_rows"], ignore_index=True) if job["trade_rows"] else pd.DataFrame()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Taraflanan Hisse", len(status_df))
    c2.metric("Başarılı", int((status_df["status"] == "ok").sum()) if not status_df.empty else 0)
    c3.metric("Fold", len(fold_df))
    c4.metric("OOS Trade", len(trade_df))

    st.subheader("Hisse Durumu")
    st.dataframe(status_df, use_container_width=True)

    if not fold_df.empty:
        st.subheader("Walk-forward Fold Sonuçları")
        st.dataframe(fold_df, use_container_width=True)

    if not train_df.empty:
        st.subheader("Eğitim Parametre Özeti")
        st.dataframe(train_df, use_container_width=True)

    if not trade_df.empty:
        st.subheader("OOS Trade Kayıtları")
        st.dataframe(trade_df, use_container_width=True)

    st.download_button(
        "Symbol status CSV indir",
        data=status_df.to_csv(index=False).encode("utf-8"),
        file_name="symbol_status.csv",
        mime="text/csv",
    )
    if not fold_df.empty:
        st.download_button(
            "Fold CSV indir",
            data=fold_df.to_csv(index=False).encode("utf-8"),
            file_name="walk_forward_folds.csv",
            mime="text/csv",
        )
    if not trade_df.empty:
        st.download_button(
            "Trade CSV indir",
            data=trade_df.to_csv(index=False).encode("utf-8"),
            file_name="oos_trades.csv",
            mime="text/csv",
        )
else:
    st.info("Sol menüden ayarları belirleyip **Backtest başlat / yenile** butonuna basın. İşlem parça parça ilerler.")

results_path = Path("results")
if results_path.exists():
    st.caption(f"Son kayıt klasörü: {results_path.resolve()}")
