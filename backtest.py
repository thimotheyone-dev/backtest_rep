from __future__ import annotations

import argparse
import itertools
import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import yfinance as yf
from concurrent.futures import ThreadPoolExecutor, as_completed


@dataclass(frozen=True)
class Params:
    ma_fast: int = 50
    ma_slow: int = 200
    rsi_period: int = 14
    rsi_low: float = 55.0
    rsi_high: float = 70.0
    rel_vol_threshold: float = 1.5
    adx_threshold: float = 20.0
    breakout_lookback: int = 20
    hold_days: int = 10
    stop_loss_pct: float = 0.05
    take_profit_pct: float = 0.10


@dataclass
class Trade:
    symbol: str
    entry_date: str
    exit_date: str
    entry_price: float
    exit_price: float
    return_pct: float
    bars_held: int
    exit_reason: str
    params: dict


def read_symbols(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"Symbol list not found: {path}")
    symbols = []
    for line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        s = line.strip().upper()
        if s:
            symbols.append(s)
    seen = set()
    out = []
    for s in symbols:
        if s not in seen:
            seen.add(s)
            out.append(s)
    return out


def to_yfinance_symbol(symbol: str) -> str:
    s = symbol.strip().upper()
    return s if s.endswith(".IS") else f"{s}.IS"


def flatten_col(col) -> str:
    if isinstance(col, tuple):
        col = "_".join(str(x) for x in col if x not in (None, "", "nan"))
    else:
        col = str(col)
    col = col.strip()
    if col in {"Adj Close", "Adj_Close"}:
        return "Adj Close"
    for key in ["Date", "Datetime", "index", "Open", "High", "Low", "Close", "Volume", "Adj Close"]:
        if col == key:
            return "Date" if key in {"Date", "Datetime", "index"} else key
        if col.startswith(key + "_"):
            return "Date" if key in {"Date", "Datetime", "index"} else key
    return col


def normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Date" not in out.columns:
        out = out.reset_index()
        out.columns = [flatten_col(c) for c in out.columns]
        if "Date" not in out.columns:
            out = out.rename(columns={out.columns[0]: "Date"})
    rename_map = {}
    for c in out.columns:
        if c.startswith("Open_"):
            rename_map[c] = "Open"
        elif c.startswith("High_"):
            rename_map[c] = "High"
        elif c.startswith("Low_"):
            rename_map[c] = "Low"
        elif c.startswith("Close_"):
            rename_map[c] = "Close"
        elif c.startswith("Adj Close_"):
            rename_map[c] = "Adj Close"
        elif c.startswith("Volume_"):
            rename_map[c] = "Volume"
    if rename_map:
        out = out.rename(columns=rename_map)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    for c in ["Open", "High", "Low", "Close", "Volume"]:
        if c in out.columns:
            out[c] = pd.to_numeric(out[c], errors="coerce")
    out = out.dropna(subset=["Date", "Open", "High", "Low", "Close", "Volume"]).copy()
    out = out.sort_values("Date").drop_duplicates("Date").reset_index(drop=True)
    return out


def load_ohlcv(symbol: str, period: str, interval: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    ticker = to_yfinance_symbol(symbol)
    cache_path = None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{ticker.replace('.', '_')}_{period}_{interval}.csv"
        if cache_path.exists():
            try:
                cached = pd.read_csv(cache_path, parse_dates=["Date"])
                return normalize_ohlcv(cached)
            except Exception:
                pass

    raw = yf.download(
        ticker,
        period=period,
        interval=interval,
        auto_adjust=False,
        group_by="column",
        progress=False,
        threads=True,
    )
    if raw is None or raw.empty:
        return pd.DataFrame()
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [flatten_col(c) for c in df.columns.to_flat_index()]
    else:
        df.columns = [flatten_col(c) for c in df.columns]
    df = df.reset_index()
    df.columns = [flatten_col(c) for c in df.columns]
    if "Date" not in df.columns:
        if "Datetime" in df.columns:
            df = df.rename(columns={"Datetime": "Date"})
        elif "index" in df.columns:
            df = df.rename(columns={"index": "Date"})
        else:
            df.insert(0, "Date", pd.to_datetime(raw.index))
    df = normalize_ohlcv(df)
    if cache_path is not None and not df.empty:
        try:
            df.to_csv(cache_path, index=False)
        except Exception:
            pass
    return df


def sma(series: pd.Series, period: int) -> pd.Series:
    return series.rolling(period).mean()


def rsi(close: pd.Series, period: int = 14) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return (100 - (100 / (1 + rs))).bfill().fillna(50.0)


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high = df["High"]
    low = df["Low"]
    close = df["Close"]
    up_move = high.diff()
    down_move = -low.diff()
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)
    tr = pd.concat([(high - low), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    plus_di = 100 * pd.Series(plus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    minus_di = 100 * pd.Series(minus_dm, index=df.index).ewm(alpha=1 / period, min_periods=period, adjust=False).mean() / atr
    dx = (100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan)
    return dx.ewm(alpha=1 / period, min_periods=period, adjust=False).mean().bfill()


def relative_volume(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["Volume"] / df["Volume"].shift(1).rolling(lookback).mean()


def resistance_level(df: pd.DataFrame, lookback: int = 20) -> pd.Series:
    return df["High"].shift(1).rolling(lookback).max()


def build_features(df: pd.DataFrame, params: Params) -> pd.DataFrame:
    out = df.copy()
    out[f"MA_{params.ma_fast}"] = sma(out["Close"], params.ma_fast)
    out[f"MA_{params.ma_slow}"] = sma(out["Close"], params.ma_slow)
    out[f"RSI_{params.rsi_period}"] = rsi(out["Close"], params.rsi_period)
    out["ADX_14"] = adx(out, 14)
    out["RelVol_20"] = relative_volume(out, 20)
    out["Resistance"] = resistance_level(out, params.breakout_lookback)
    fast = out[f"MA_{params.ma_fast}"]
    slow = out[f"MA_{params.ma_slow}"]
    rsi_col = out[f"RSI_{params.rsi_period}"]
    out["signal"] = (
        (fast > slow)
        & (out["Close"] > fast)
        & (rsi_col.between(params.rsi_low, params.rsi_high))
        & (out["RelVol_20"] > params.rel_vol_threshold)
        & (out["ADX_14"] > params.adx_threshold)
        & (out["Close"] > out["Resistance"])
    ).fillna(False)
    return out


def simulate_trades(df: pd.DataFrame, symbol: str, params: Params) -> list[Trade]:
    feat = build_features(df, params)
    trades: list[Trade] = []
    n = len(feat)
    if n < max(params.ma_slow, params.breakout_lookback) + params.hold_days + 5:
        return trades

    i = 0
    while i < n - 2:
        if not bool(feat.loc[i, "signal"]):
            i += 1
            continue

        entry_idx = i + 1
        if entry_idx >= n:
            break
        entry_row = feat.iloc[entry_idx]
        entry_price = float(entry_row["Open"])
        if not np.isfinite(entry_price) or entry_price <= 0:
            i += 1
            continue

        stop_price = entry_price * (1 - params.stop_loss_pct)
        take_price = entry_price * (1 + params.take_profit_pct)

        max_exit_idx = min(entry_idx + params.hold_days, n - 1)
        exit_idx = max_exit_idx
        exit_price = float(feat.iloc[exit_idx]["Close"])
        exit_reason = f"hold_{params.hold_days}"
        exit_date = feat.iloc[exit_idx]["Date"]

        for j in range(entry_idx, max_exit_idx + 1):
            row = feat.iloc[j]
            low = float(row["Low"])
            high = float(row["High"])
            stop_hit = low <= stop_price
            take_hit = high >= take_price
            if stop_hit and take_hit:
                exit_idx = j
                exit_price = stop_price
                exit_reason = "stop_and_take_same_day_stop_first"
                exit_date = row["Date"]
                break
            if stop_hit:
                exit_idx = j
                exit_price = stop_price
                exit_reason = "stop_loss"
                exit_date = row["Date"]
                break
            if take_hit:
                exit_idx = j
                exit_price = take_price
                exit_reason = "take_profit"
                exit_date = row["Date"]
                break

        ret_pct = (exit_price / entry_price) - 1.0
        trades.append(
            Trade(
                symbol=symbol,
                entry_date=str(pd.to_datetime(entry_row["Date"]).date()),
                exit_date=str(pd.to_datetime(exit_date).date()),
                entry_price=entry_price,
                exit_price=exit_price,
                return_pct=ret_pct,
                bars_held=int(exit_idx - entry_idx + 1),
                exit_reason=exit_reason,
                params=asdict(params),
            )
        )
        i = exit_idx + 1

    return trades


def metrics_from_trades(trades: list[Trade]) -> dict:
    if not trades:
        return {
            "trades": 0,
            "win_rate": np.nan,
            "avg_return": np.nan,
            "median_return": np.nan,
            "profit_factor": np.nan,
            "expectancy": np.nan,
            "max_loss": np.nan,
            "avg_bars_held": np.nan,
            "cum_return": np.nan,
        }
    rets = np.array([t.return_pct for t in trades], dtype=float)
    wins = rets[rets > 0]
    losses = rets[rets <= 0]
    gross_profit = wins.sum()
    gross_loss = abs(losses.sum())
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else np.inf
    return {
        "trades": int(len(trades)),
        "win_rate": float((rets > 0).mean()),
        "avg_return": float(rets.mean()),
        "median_return": float(np.median(rets)),
        "profit_factor": float(profit_factor),
        "expectancy": float(rets.mean()),
        "max_loss": float(rets.min()),
        "avg_bars_held": float(np.mean([t.bars_held for t in trades])),
        "cum_return": float(np.prod(1 + rets) - 1),
    }


def score_result(metrics: dict) -> float:
    if metrics.get("trades", 0) < 20:
        return -1e18
    for key in ["win_rate", "avg_return", "profit_factor", "max_loss"]:
        if pd.isna(metrics.get(key, np.nan)):
            return -1e18
    return (
        1000 * float(metrics["avg_return"])
        + 0.8 * float(metrics["win_rate"])
        + 0.2 * min(float(metrics["profit_factor"]), 10.0)
        + 0.3 * float(metrics["max_loss"])
    )


def walk_forward_splits(n: int, train_size: int, test_size: int, step_size: int):
    start = 0
    while True:
        train_start = start
        train_end = train_start + train_size
        test_start = train_end
        test_end = test_start + test_size
        if test_end > n:
            break
        yield train_start, train_end, test_start, test_end
        start += step_size


def build_param_grid(
    ma_fast_list,
    ma_slow_list,
    rsi_period_list,
    rsi_low_list,
    rsi_high_list,
    rel_vol_list,
    adx_list,
    breakout_list,
    hold_days_list,
    stop_loss_list,
    take_profit_list,
) -> list[Params]:
    grid = []
    for combo in itertools.product(
        ma_fast_list,
        ma_slow_list,
        rsi_period_list,
        rsi_low_list,
        rsi_high_list,
        rel_vol_list,
        adx_list,
        breakout_list,
        hold_days_list,
        stop_loss_list,
        take_profit_list,
    ):
        ma_fast, ma_slow, rsi_period, rsi_low, rsi_high, rel_vol, adx_th, brk, hold_days, sl, tp = combo
        if ma_fast >= ma_slow or rsi_low >= rsi_high:
            continue
        grid.append(
            Params(
                ma_fast=ma_fast,
                ma_slow=ma_slow,
                rsi_period=rsi_period,
                rsi_low=rsi_low,
                rsi_high=rsi_high,
                rel_vol_threshold=rel_vol,
                adx_threshold=adx_th,
                breakout_lookback=brk,
                hold_days=hold_days,
                stop_loss_pct=sl,
                take_profit_pct=tp,
            )
        )
    return grid


def optimize_on_train(df: pd.DataFrame, symbol: str, param_grid: list[Params]):
    rows = []
    best_score = -1e18
    best_params = None
    for p in param_grid:
        trades = simulate_trades(df, symbol, p)
        metrics = metrics_from_trades(trades)
        metrics["score"] = score_result(metrics)
        metrics["symbol"] = symbol
        metrics["params"] = json.dumps(asdict(p), ensure_ascii=False)
        rows.append(metrics)
        if metrics["score"] > best_score:
            best_score = metrics["score"]
            best_params = p
    return best_params, pd.DataFrame(rows)


def walk_forward_backtest(df: pd.DataFrame, symbol: str, param_grid: list[Params], train_size: int, test_size: int, step_size: int):
    fold_rows = []
    oos_trade_rows = []
    splits = list(walk_forward_splits(len(df), train_size, test_size, step_size))
    if not splits:
        return pd.DataFrame(), pd.DataFrame()

    for fold_id, (tr_start, tr_end, te_start, te_end) in enumerate(splits, start=1):
        train_df = df.iloc[tr_start:tr_end].reset_index(drop=True)
        test_df = df.iloc[te_start:te_end].reset_index(drop=True)
        best_params, _ = optimize_on_train(train_df, symbol, param_grid)
        if best_params is None:
            continue
        test_trades = simulate_trades(test_df, symbol, best_params)
        metrics = metrics_from_trades(test_trades)
        metrics["score"] = score_result(metrics)
        metrics["symbol"] = symbol
        metrics["fold"] = fold_id
        metrics["train_start"] = str(pd.to_datetime(train_df.iloc[0]["Date"]).date())
        metrics["train_end"] = str(pd.to_datetime(train_df.iloc[-1]["Date"]).date())
        metrics["test_start"] = str(pd.to_datetime(test_df.iloc[0]["Date"]).date())
        metrics["test_end"] = str(pd.to_datetime(test_df.iloc[-1]["Date"]).date())
        metrics["best_params"] = json.dumps(asdict(best_params), ensure_ascii=False)
        fold_rows.append(metrics)

        for t in test_trades:
            d = asdict(t)
            d["fold"] = fold_id
            oos_trade_rows.append(d)

    return pd.DataFrame(fold_rows), pd.DataFrame(oos_trade_rows)


def process_symbol(symbol: str, period: str, interval: str, cache_dir: Path, param_grid: list[Params], train_size: int, test_size: int, step_size: int):
    df = load_ohlcv(symbol, period, interval, cache_dir=cache_dir)
    if df.empty or len(df) < 300:
        return symbol, {
            "status": "no_data_or_too_short",
            "fold_summary": pd.DataFrame(),
            "train_summary": pd.DataFrame(),
            "oos_trades": pd.DataFrame(),
        }

    fold_summary, oos_trades = walk_forward_backtest(df, symbol, param_grid, train_size, test_size, step_size)
    if fold_summary.empty:
        return symbol, {
            "status": "no_folds",
            "fold_summary": fold_summary,
            "train_summary": pd.DataFrame(),
            "oos_trades": oos_trades,
        }

    grouped = fold_summary.groupby("best_params", as_index=False).agg(
        folds=("fold", "count"),
        score_mean=("score", "mean"),
        score_median=("score", "median"),
        trades_mean=("trades", "mean"),
        win_rate_mean=("win_rate", "mean"),
        avg_return_mean=("avg_return", "mean"),
        pf_mean=("profit_factor", "mean"),
        cum_return_mean=("cum_return", "mean"),
        max_loss_mean=("max_loss", "mean"),
    ).sort_values(["score_mean", "cum_return_mean"], ascending=False)

    return symbol, {
        "status": "ok",
        "fold_summary": fold_summary,
        "train_summary": grouped,
        "oos_trades": oos_trades,
    }


def main():
    parser = argparse.ArgumentParser(description="BIST indicator optimization backtest")
    parser.add_argument("--symbols-file", type=str, default="bist_tum.txt")
    parser.add_argument("--outdir", type=str, default="backtest_output")
    parser.add_argument("--cache-dir", type=str, default="cache")
    parser.add_argument("--period", type=str, default="10y")
    parser.add_argument("--interval", type=str, default="1d")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--train-size", type=int, default=504)
    parser.add_argument("--test-size", type=int, default=126)
    parser.add_argument("--step-size", type=int, default=126)
    parser.add_argument("--max-symbols", type=int, default=0)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    symbols = read_symbols(Path(args.symbols_file))
    if args.max_symbols and args.max_symbols > 0:
        symbols = symbols[: args.max_symbols]

    if args.quick:
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

    print(f"Symbols: {len(symbols)}")
    print(f"Param combinations: {len(param_grid)}")

    status_rows = []
    train_summaries = []
    fold_summaries = []
    oos_trades_all = []

    t0 = time.time()
    with ThreadPoolExecutor(max_workers=args.workers) as ex:
        futures = {
            ex.submit(process_symbol, s, args.period, args.interval, cache_dir, param_grid, args.train_size, args.test_size, args.step_size): s
            for s in symbols
        }
        for fut in as_completed(futures):
            symbol = futures[fut]
            try:
                _, result = fut.result()
            except Exception as e:
                print(f"[{symbol}] ERROR: {e}")
                continue

            status_rows.append({
                "symbol": symbol,
                "status": result["status"],
            })

            if not result["train_summary"].empty:
                ts = result["train_summary"].copy()
                ts.insert(0, "symbol", symbol)
                train_summaries.append(ts)

            if not result["fold_summary"].empty:
                fs = result["fold_summary"].copy()
                fs.insert(0, "symbol", symbol)
                fold_summaries.append(fs)

            if not result["oos_trades"].empty:
                ot = result["oos_trades"].copy()
                ot.insert(0, "symbol", symbol)
                oos_trades_all.append(ot)

            print(f"[{symbol}] done")

    status_df = pd.DataFrame(status_rows)
    train_df = pd.concat(train_summaries, ignore_index=True) if train_summaries else pd.DataFrame()
    fold_df = pd.concat(fold_summaries, ignore_index=True) if fold_summaries else pd.DataFrame()
    trades_df = pd.concat(oos_trades_all, ignore_index=True) if oos_trades_all else pd.DataFrame()

    status_df.to_csv(outdir / "symbol_status.csv", index=False)
    if not train_df.empty:
        train_df.to_csv(outdir / "training_parameter_summary.csv", index=False)
    if not fold_df.empty:
        fold_df.to_csv(outdir / "walk_forward_folds.csv", index=False)
    if not trades_df.empty:
        trades_df.to_csv(outdir / "oos_trades.csv", index=False)

    if not fold_df.empty:
        best = fold_df.groupby("best_params", as_index=False).agg(
            folds=("fold", "count"),
            score_mean=("score", "mean"),
            score_median=("score", "median"),
            trades_mean=("trades", "mean"),
            win_rate_mean=("win_rate", "mean"),
            avg_return_mean=("avg_return", "mean"),
            pf_mean=("profit_factor", "mean"),
            cum_return_mean=("cum_return", "mean"),
            max_loss_mean=("max_loss", "mean"),
        ).sort_values(["score_mean", "cum_return_mean"], ascending=False)
        best.to_csv(outdir / "best_parameter_sets.csv", index=False)
        top = best.head(20).to_dict("records")
    else:
        top = []

    elapsed = time.time() - t0
    with open(outdir / "run_summary.json", "w", encoding="utf-8") as f:
        json.dump({
            "symbols": len(symbols),
            "parameter_combinations": len(param_grid),
            "elapsed_seconds": elapsed,
            "quick_mode": bool(args.quick),
            "top_parameter_sets": top,
        }, f, ensure_ascii=False, indent=2)

    print("\nDone.")
    print(f"Elapsed: {elapsed:.1f}s")
    print(f"Output: {outdir.resolve()}")


if __name__ == "__main__":
    main()
