"""Train counterparty-flow Ridge models OFFLINE on Round 4 data.

Pipeline
--------
1. Load all 3 days of trades + prices (semicolon CSVs).
2. Filter block trades (buyer in {Mark 01, Mark 67} AND seller in {Mark 22, Mark 49}).
3. Build per-trade features: buyer/seller one-hot, log-qty, direction sign,
   rolling counterparty flows (5, 20, 50 bars), voucher moneyness.
4. Target: forward log-return at horizons 5, 20, 50 bars (1 bar = 100 ts).
5. Train Ridge (alpha=10) per (symbol, horizon). Validate on day 3 (train days 1+2).
6. Print IC, hit-rate, implied Sharpe per (symbol, horizon).
7. Export weights + standardization params to /Users/emmett/Documents/Round4/cp_alpha.json.

The inference helper /Users/emmett/Documents/Round4/cp_features.py mirrors the
feature-extraction logic in pure stdlib for use inside trader.py.
"""

from __future__ import annotations

import json
import math
import os
import sys
from collections import defaultdict, deque

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

# Local helper (also used by trader.py at inference time)
import cp_features as cpf

DATA_DIR = "/Users/emmett/Documents/Round4/ROUND_4"
OUT_JSON = "/Users/emmett/Documents/Round4/cp_alpha.json"
OUT_REPORT = "/Users/emmett/Documents/Round4/cp_alpha_report.md"

DAYS_TRAIN = (1, 2)
DAY_TEST = 3

BAR = 100  # one bar = 100 timestamp units
HORIZONS_BARS = (5, 20, 50)
RIDGE_ALPHA = 10.0

# Time-to-expiry: vouchers expire in N rounds. We treat each day as ~1/252 yr.
# For a relative measure, T = (TOTAL_DAYS_LEFT - elapsed_day_fraction) / 252.
# At round 4 there are typically ~3 trading days left; we use TOTAL_DAYS_LEFT = 3.
TOTAL_DAYS_LEFT = 3.0


# --------------------------------------------------------------------------- #
# Loading
# --------------------------------------------------------------------------- #
def load_day(day: int):
    trades = pd.read_csv(f"{DATA_DIR}/trades_round_4_day_{day}.csv", sep=";")
    prices = pd.read_csv(f"{DATA_DIR}/prices_round_4_day_{day}.csv", sep=";")
    trades["day"] = day
    if "day" not in prices.columns:
        prices["day"] = day
    return trades, prices


def load_all():
    all_t, all_p = [], []
    for d in (1, 2, 3):
        t, p = load_day(d)
        all_t.append(t)
        all_p.append(p)
    return pd.concat(all_t, ignore_index=True), pd.concat(all_p, ignore_index=True)


# --------------------------------------------------------------------------- #
# Mid-price index per (day, product)
# --------------------------------------------------------------------------- #
def build_mid_lookups(prices: pd.DataFrame):
    """Return dict {(day, product): (sorted_ts_array, mid_array)} for fast lookup."""
    out = {}
    for (day, prod), grp in prices.groupby(["day", "product"]):
        g = grp.sort_values("timestamp")
        out[(day, prod)] = (g["timestamp"].to_numpy(), g["mid_price"].to_numpy(dtype=float))
    return out


def mid_at(mid_idx, day, product, ts):
    key = (day, product)
    if key not in mid_idx:
        return np.nan
    arr_ts, arr_mid = mid_idx[key]
    # right-side search: latest mid with timestamp <= ts
    i = np.searchsorted(arr_ts, ts, side="right") - 1
    if i < 0:
        return np.nan
    return float(arr_mid[i])


def mid_at_or_after(mid_idx, day, product, ts):
    key = (day, product)
    if key not in mid_idx:
        return np.nan
    arr_ts, arr_mid = mid_idx[key]
    i = np.searchsorted(arr_ts, ts, side="left")
    if i >= len(arr_ts):
        return np.nan
    return float(arr_mid[i])


# --------------------------------------------------------------------------- #
# Feature build
# --------------------------------------------------------------------------- #
def build_features_for_day(trades_day: pd.DataFrame, prices_day: pd.DataFrame,
                           mid_idx, day_value: int):
    """Return dict {symbol: (X_list, y_dict_h: list, ts_list, meta...)}.

    We process trades chronologically (across all symbols) to keep one global
    flow-tracker so each trade's recent-flow features incorporate every other
    counterparty's prior activity (block trades excluded).
    """
    # filter and sort
    df = trades_day.sort_values(["timestamp"]).reset_index(drop=True)

    flow_tracker = cpf.FlowTracker(windows_bars=cpf.ROLLING_WINDOWS_BARS, bar_size=BAR)

    samples_by_sym = defaultdict(lambda: {"X": [], "ts": [], "y": {h: [] for h in HORIZONS_BARS}})

    for row in df.itertuples(index=False):
        ts = int(row.timestamp)
        buyer = row.buyer
        seller = row.seller
        sym = row.symbol
        qty = float(row.quantity)
        price = float(row.price)

        is_block = cpf.is_block_trade(buyer, seller)

        # current symbol mid
        mid = mid_at(mid_idx, day_value, sym, ts)

        # spot for moneyness
        spot = mid_at(mid_idx, day_value, "VELVETFRUIT_EXTRACT", ts)

        # T: assume each "day" has timestamps in [0, ~1e6]; fractional progress through day
        # day fraction in [0,1]; use elapsed days = (day_value - 1) + ts / 1e6
        elapsed = (day_value - 1) + ts / 1_000_000.0
        T_years = max(1e-4, (TOTAL_DAYS_LEFT - elapsed) / 252.0)

        # snapshot recent flows BEFORE updating (so we don't leak the current trade)
        recent = flow_tracker.snapshot(ts)

        # update tracker (skips block trades internally)
        flow_tracker.update(ts, buyer, seller, qty)

        if is_block:
            continue
        if mid is None or not np.isfinite(mid) or mid <= 0:
            continue

        feats = cpf.extract_trade_features(buyer, seller, qty, price, mid, sym,
                                            recent, spot=spot, T=T_years)

        # forward returns at horizons
        ys = {}
        ok = True
        for h in HORIZONS_BARS:
            future_ts = ts + h * BAR
            future_mid = mid_at_or_after(mid_idx, day_value, sym, future_ts)
            if future_mid is None or not np.isfinite(future_mid) or future_mid <= 0:
                ok = False
                break
            ys[h] = math.log(future_mid / mid)
        if not ok:
            continue

        rec = samples_by_sym[sym]
        rec["X"].append(feats)
        rec["ts"].append(ts)
        for h in HORIZONS_BARS:
            rec["y"][h].append(ys[h])

    return samples_by_sym


def merge_samples(samples_list):
    """Merge multiple per-day sample dicts into one."""
    out = defaultdict(lambda: {"X": [], "ts": [], "y": {h: [] for h in HORIZONS_BARS}})
    for s in samples_list:
        for sym, rec in s.items():
            out[sym]["X"].extend(rec["X"])
            out[sym]["ts"].extend(rec["ts"])
            for h in HORIZONS_BARS:
                out[sym]["y"][h].extend(rec["y"][h])
    return out


# --------------------------------------------------------------------------- #
# Training
# --------------------------------------------------------------------------- #
def fit_eval_symbol(sym: str, train_rec, test_rec):
    feat_names = cpf.feature_names(sym)
    p = len(feat_names)
    Xtr = np.asarray(train_rec["X"], dtype=float)
    Xte = np.asarray(test_rec["X"], dtype=float) if test_rec["X"] else np.zeros((0, p))

    if Xtr.shape[0] == 0:
        return None
    if Xtr.shape[1] != p:
        # safety: pad / truncate (shouldn't happen)
        return None

    # standardize using TRAIN stats only
    mu = Xtr.mean(axis=0)
    sd = Xtr.std(axis=0)
    sd_safe = np.where(sd > 1e-8, sd, 1.0)
    Xtr_s = (Xtr - mu) / sd_safe
    if Xte.shape[0] > 0:
        Xte_s = (Xte - mu) / sd_safe
    else:
        Xte_s = Xte

    results = {}
    for h in HORIZONS_BARS:
        ytr = np.asarray(train_rec["y"][h], dtype=float)
        yte = np.asarray(test_rec["y"][h], dtype=float) if test_rec["y"][h] else np.zeros(0)
        if ytr.size == 0:
            continue
        model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
        model.fit(Xtr_s, ytr)
        coefs = model.coef_.tolist()
        intercept = float(model.intercept_)

        # in-sample
        pred_tr = model.predict(Xtr_s)
        ic_tr = _safe_corr(pred_tr, ytr)

        # out-of-sample
        if Xte_s.shape[0] > 0 and yte.size > 0:
            pred_te = model.predict(Xte_s)
            ic_te = _safe_corr(pred_te, yte)
            # hit rate: sign agreement (excluding pred==0)
            mask = pred_te != 0
            hit = float((np.sign(pred_te[mask]) == np.sign(yte[mask])).mean()) if mask.any() else float("nan")
            # implied sharpe of pred-weighted strategy: mean(pred*y)/std(pred*y)*sqrt(N_per_day)
            pnl = pred_te * yte
            if pnl.std() > 0:
                # rough: trades per day ~ len(yte)/3? We just report annualization-free Sharpe:
                sharpe = float(pnl.mean() / pnl.std()) * math.sqrt(max(1, len(yte)))
            else:
                sharpe = float("nan")
        else:
            ic_te = float("nan")
            hit = float("nan")
            sharpe = float("nan")
            pred_te = np.zeros(0)

        results[h] = {
            "feature_names": feat_names,
            "means": mu.tolist(),
            "stds": sd_safe.tolist(),
            "coefs": coefs,
            "intercept": intercept,
            "n_train": int(Xtr.shape[0]),
            "n_test": int(Xte.shape[0]),
            "ic_train": float(ic_tr) if np.isfinite(ic_tr) else None,
            "ic_test": float(ic_te) if np.isfinite(ic_te) else None,
            "hit_rate_test": float(hit) if np.isfinite(hit) else None,
            "sharpe_test": float(sharpe) if np.isfinite(sharpe) else None,
        }
    return results


def _safe_corr(a, b):
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    if a.size < 3:
        return float("nan")
    sa = a.std(); sb = b.std()
    if sa <= 1e-12 or sb <= 1e-12:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    print("Loading data...")
    trades_all, prices_all = load_all()
    mid_idx = build_mid_lookups(prices_all)
    print(f"  trades: {len(trades_all)} rows; prices: {len(prices_all)} rows")

    # per-day feature build
    per_day = {}
    for d in (1, 2, 3):
        td = trades_all[trades_all["day"] == d]
        pd_ = prices_all[prices_all["day"] == d]
        print(f"Building features for day {d}: {len(td)} trades, {len(pd_)} price ticks")
        per_day[d] = build_features_for_day(td, pd_, mid_idx, d)

    train_samples = merge_samples([per_day[d] for d in DAYS_TRAIN])
    test_samples = per_day[DAY_TEST]

    print("\nTraining Ridge models per (symbol, horizon)...")
    all_models = {}
    rows = []  # for the report
    for sym in sorted(set(list(train_samples.keys()) + list(test_samples.keys()))):
        tr = train_samples.get(sym, {"X": [], "ts": [], "y": {h: [] for h in HORIZONS_BARS}})
        te = test_samples.get(sym, {"X": [], "ts": [], "y": {h: [] for h in HORIZONS_BARS}})
        if not tr["X"]:
            continue
        res = fit_eval_symbol(sym, tr, te)
        if res is None:
            continue
        all_models[sym] = res
        for h, info in res.items():
            rows.append({
                "symbol": sym,
                "horizon_bars": h,
                "n_train": info["n_train"],
                "n_test": info["n_test"],
                "ic_train": info["ic_train"],
                "ic_test": info["ic_test"],
                "hit_rate_test": info["hit_rate_test"],
                "sharpe_test": info["sharpe_test"],
            })
            print(f"  {sym:22s} h={h:>3} bars  n_tr={info['n_train']:5d} n_te={info['n_test']:5d}  "
                  f"IC_tr={_fmt(info['ic_train'])}  IC_te={_fmt(info['ic_test'])}  "
                  f"hit={_fmt(info['hit_rate_test'])}  Sharpe={_fmt(info['sharpe_test'])}")

    # Top-feature analysis: pick the single best (symbol, horizon) by ic_test, magnitude of coefs
    best = None
    for r in rows:
        if r["ic_test"] is None:
            continue
        if best is None or r["ic_test"] > best["ic_test"]:
            best = r
    top_feats_summary = ""
    if best is not None:
        info = all_models[best["symbol"]][best["horizon_bars"]]
        coefs = np.asarray(info["coefs"])
        idx = np.argsort(-np.abs(coefs))[:5]
        top = [(info["feature_names"][i], float(coefs[i])) for i in idx]
        top_feats_summary = f"\nTop-5 |coef| features for best model {best['symbol']} h={best['horizon_bars']}:\n"
        for n, c in top:
            top_feats_summary += f"  {n:30s}  {c:+.6f}\n"
        print(top_feats_summary)

    # Save model JSON
    payload = {
        "schema": "cp_alpha_v1",
        "bar": BAR,
        "horizons_bars": list(HORIZONS_BARS),
        "ridge_alpha": RIDGE_ALPHA,
        "marks": cpf.MARKS,
        "windows_bars": list(cpf.ROLLING_WINDOWS_BARS),
        "voucher_strikes": cpf.VOUCHER_STRIKES,
        "models": {},
    }
    for sym, by_h in all_models.items():
        payload["models"][sym] = {}
        for h, info in by_h.items():
            payload["models"][sym][str(h)] = {
                "feature_names": info["feature_names"],
                "means": info["means"],
                "stds": info["stds"],
                "coefs": info["coefs"],
                "intercept": info["intercept"],
                "ic_train": info["ic_train"],
                "ic_test": info["ic_test"],
                "hit_rate_test": info["hit_rate_test"],
                "sharpe_test": info["sharpe_test"],
                "n_train": info["n_train"],
                "n_test": info["n_test"],
            }
    with open(OUT_JSON, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nWrote model file: {OUT_JSON}")

    # Report
    rows_sorted = sorted(rows, key=lambda r: (r["symbol"], r["horizon_bars"]))
    md = ["# cp_alpha validation report", "",
          f"- Train days: {DAYS_TRAIN}; Test day: {DAY_TEST}",
          f"- Ridge alpha: {RIDGE_ALPHA}",
          f"- Bar size: {BAR}; Horizons: {HORIZONS_BARS}",
          f"- Block-trade filter: buyer in {sorted(cpf.BLOCK_BUYERS)} AND seller in {sorted(cpf.BLOCK_SELLERS)}",
          "",
          "| Symbol | Horizon | n_train | n_test | IC_train | IC_test | Hit | Sharpe |",
          "|---|---:|---:|---:|---:|---:|---:|---:|"]
    for r in rows_sorted:
        md.append(f"| {r['symbol']} | {r['horizon_bars']} | {r['n_train']} | {r['n_test']} "
                  f"| {_fmt(r['ic_train'])} | {_fmt(r['ic_test'])} "
                  f"| {_fmt(r['hit_rate_test'])} | {_fmt(r['sharpe_test'])} |")

    # Significance: at least one (sym, h) with IC_test>0.10 AND Sharpe_test>0.10
    sig = [r for r in rows
           if r["ic_test"] is not None and r["sharpe_test"] is not None
           and r["ic_test"] > 0.10 and r["sharpe_test"] > 0.10]
    md.append("")
    md.append(f"**Significant (IC>0.10 AND Sharpe>0.10) on day-3 holdout: {len(sig)}**")
    if sig:
        md.append("")
        md.append("| Symbol | Horizon | IC_test | Sharpe |")
        md.append("|---|---:|---:|---:|")
        for r in sig:
            md.append(f"| {r['symbol']} | {r['horizon_bars']} | {_fmt(r['ic_test'])} | {_fmt(r['sharpe_test'])} |")
    if best is not None:
        md.append("")
        md.append(top_feats_summary)
    md.append("")
    md.append(f"**Recommend integration into trader.py: {'YES' if sig else 'NO'}**")

    with open(OUT_REPORT, "w") as f:
        f.write("\n".join(md))
    print(f"Wrote report: {OUT_REPORT}")
    print(f"\nSignificant models: {len(sig)}")
    return 0


def _fmt(x):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "n/a"
    return f"{x:+.4f}"


if __name__ == "__main__":
    sys.exit(main())
