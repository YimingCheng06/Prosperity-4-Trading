"""Round 4 counterparty ML alpha discovery.

Goal: identify which counterparty (Mark XX) actions, in which contexts, predict
future mid returns. Output → distilled rules for trader_round4 (no future fns).

Pipeline:
  1. Load R4 prices + trades (with buyer/seller).
  2. Per (product, timestamp), engineer features:
     - microstructure: I_top, microprice deviation, spread, recent return
     - counterparty: per-actor net flow over windows W ∈ {5,20,50}
     - this-tick triggers: (actor, role) one-hots × qty
  3. Target: mid_{t+k} - mid_t for k ∈ {5, 10, 25, 50}.
  4. Train HistGradientBoostingRegressor.
  5. Permutation importance + per-feature partial effect → rules.

CV: train on day1+day2, test on day3 (most distribution-shifted day).
"""
import pandas as pd
import numpy as np
from pathlib import Path
import sys

DATA_DIR = Path("/Users/emmett/Documents/Prosperity Trading/ROUND_4")

# ---------- 1. load ----------
def load_data():
    prices = pd.concat([
        pd.read_csv(DATA_DIR / f"prices_round_4_day_{d}.csv", sep=";").assign(dayfile=d)
        for d in [1, 2, 3]
    ], ignore_index=True)
    trades = pd.concat([
        pd.read_csv(DATA_DIR / f"trades_round_4_day_{d}.csv", sep=";").assign(dayfile=d)
        for d in [1, 2, 3]
    ], ignore_index=True)
    prices["mid_price"] = pd.to_numeric(prices["mid_price"], errors="coerce")
    prices = prices.sort_values(["dayfile", "product", "timestamp"]).reset_index(drop=True)
    trades = trades.sort_values(["dayfile", "timestamp"]).reset_index(drop=True)
    return prices, trades


# Top counterparties (>=100 trades total in any product)
ACTORS = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]


# ---------- 2. feature engineering ----------
def build_features(prices_p, trades_p, product):
    """For one (day, product), build per-timestamp feature row."""
    df = prices_p.copy()
    df["bb"] = pd.to_numeric(df["bid_price_1"], errors="coerce")
    df["ba"] = pd.to_numeric(df["ask_price_1"], errors="coerce")
    df["bv"] = pd.to_numeric(df["bid_volume_1"], errors="coerce").fillna(0)
    df["av"] = pd.to_numeric(df["ask_volume_1"], errors="coerce").fillna(0)
    df["spread"] = (df["ba"] - df["bb"])
    # microprice (size-weighted), and deviation from raw mid
    denom = (df["bv"] + df["av"]).replace(0, np.nan)
    df["microprice"] = (df["bb"] * df["av"] + df["ba"] * df["bv"]) / denom
    df["mid_dev"] = df["microprice"] - df["mid_price"]
    df["I_top"] = (df["bv"] - df["av"]) / denom
    # recent returns at various lags
    for lag in [1, 5, 20]:
        df[f"ret_{lag}"] = df["mid_price"].diff(lag)
    # rolling realized vol
    df["rvol_20"] = df["mid_price"].diff().rolling(20).std()

    # ---- counterparty features ----
    # Filter trades for this product
    tp = trades_p[trades_p["symbol"] == product].copy()
    # signed qty: +q if Mark X bought, -q if Mark X sold
    for actor in ACTORS:
        tp[f"buy_{actor}"] = np.where(tp["buyer"] == actor, tp["quantity"], 0)
        tp[f"sell_{actor}"] = np.where(tp["seller"] == actor, tp["quantity"], 0)
    # also: just-this-tick presence flags
    actor_cols = []
    for actor in ACTORS:
        actor_cols.append(f"buy_{actor}")
        actor_cols.append(f"sell_{actor}")
    if len(tp) == 0:
        # No trades for this product; create dummy zero columns
        agg = pd.DataFrame({"timestamp": df["timestamp"]})
        for c in actor_cols:
            agg[c] = 0
    else:
        agg = tp.groupby("timestamp")[actor_cols].sum().reset_index()

    # Merge per-tick aggregates
    df = df.merge(agg, on="timestamp", how="left")
    for c in actor_cols:
        df[c] = df[c].fillna(0)

    # Rolling sum windows for each actor (BEFORE current tick — shift to avoid leak)
    for W in [5, 20, 50]:
        for actor in ACTORS:
            # net flow = buys - sells, shifted by 1 so feature at t uses t-W..t-1 (no leak)
            net = df[f"buy_{actor}"] - df[f"sell_{actor}"]
            df[f"netflow_{actor}_W{W}"] = net.rolling(W, min_periods=1).sum().shift(1)

    # ---- targets: future mid_t+k - mid_t ----
    for k in [5, 10, 25, 50]:
        df[f"y_{k}"] = df["mid_price"].shift(-k) - df["mid_price"]

    return df


# ---------- 3. assemble & train ----------
def feature_cols(df):
    base = ["spread", "mid_dev", "I_top", "ret_1", "ret_5", "ret_20", "rvol_20"]
    cp = []
    for actor in ACTORS:
        cp.append(f"buy_{actor}")
        cp.append(f"sell_{actor}")
    for W in [5, 20, 50]:
        for actor in ACTORS:
            cp.append(f"netflow_{actor}_W{W}")
    return base, cp


def run_for_product(product, prices, trades):
    print(f"\n{'='*60}\n  PRODUCT: {product}\n{'='*60}")
    parts = []
    for d in [1, 2, 3]:
        p = prices[(prices["dayfile"] == d) & (prices["product"] == product)]
        t = trades[trades["dayfile"] == d]
        if len(p) == 0:
            continue
        feat = build_features(p, t, product)
        feat["dayfile"] = d
        parts.append(feat)
    df = pd.concat(parts, ignore_index=True)

    base_cols, cp_cols = feature_cols(df)
    all_feats = base_cols + cp_cols

    # drop rows with NaN target or feature
    horizons = [5, 10, 25, 50]
    target_cols = [f"y_{k}" for k in horizons]

    df_clean = df.dropna(subset=all_feats + target_cols).reset_index(drop=True)
    if len(df_clean) < 1000:
        print(f"  Too few rows ({len(df_clean)}) — skipping.")
        return None

    train = df_clean[df_clean["dayfile"].isin([1, 2])]
    test = df_clean[df_clean["dayfile"] == 3]
    print(f"  train rows: {len(train)}  test rows: {len(test)}")

    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import r2_score
    from sklearn.inspection import permutation_importance

    out_rows = []

    for k in horizons:
        y_train = train[f"y_{k}"].values
        y_test = test[f"y_{k}"].values
        X_train = train[all_feats].values
        X_test = test[all_feats].values

        m = HistGradientBoostingRegressor(
            max_iter=200, max_depth=5, learning_rate=0.05,
            l2_regularization=1.0, random_state=0,
        )
        m.fit(X_train, y_train)
        r2_tr = r2_score(y_train, m.predict(X_train))
        r2_te = r2_score(y_test, m.predict(X_test))

        # Permutation importance ON TEST (alpha that holds out-of-sample)
        # To save time, only compute on a 5K sample
        if len(X_test) > 5000:
            idx = np.random.RandomState(0).choice(len(X_test), 5000, replace=False)
            X_imp, y_imp = X_test[idx], y_test[idx]
        else:
            X_imp, y_imp = X_test, y_test
        pi = permutation_importance(m, X_imp, y_imp, n_repeats=3, random_state=0, n_jobs=1)
        imp = pd.Series(pi.importances_mean, index=all_feats).sort_values(ascending=False)

        print(f"\n  k={k}:  train R²={r2_tr:.4f}  test R²={r2_te:.4f}")
        print(f"  Top 12 feature permutation importance (test):")
        for fname in imp.head(12).index:
            print(f"    {fname:<35s} {imp[fname]:+.5f}")

        for fname, val in imp.items():
            out_rows.append({
                "product": product, "k": k, "feature": fname, "perm_imp": val,
                "r2_test": r2_te,
            })

    return pd.DataFrame(out_rows)


# ---------- main ----------
if __name__ == "__main__":
    prices, trades = load_data()
    print(f"Loaded {len(prices)} price rows, {len(trades)} trade rows.")

    # Most interesting products: HYDROGEL, VFE, VEV_5000-5200 (most counterparty diversity)
    products = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT",
                "VEV_4000", "VEV_5000", "VEV_5100", "VEV_5200", "VEV_5300"]
    all_imp = []
    for p in products:
        r = run_for_product(p, prices, trades)
        if r is not None:
            all_imp.append(r)

    big = pd.concat(all_imp, ignore_index=True) if all_imp else None
    if big is not None:
        big.to_csv("ml_counterparty_imp.csv", index=False)
        print("\n\nWrote ml_counterparty_imp.csv")

        # Summary: average importance of counterparty features vs baseline features
        is_cp = big["feature"].str.startswith(("buy_", "sell_", "netflow_"))
        is_base = ~is_cp
        print("\n=== Average permutation importance by feature class ===")
        print("                   product           k   class  avg_imp")
        for (p, k), grp in big.groupby(["product", "k"]):
            cp_imp = grp[is_cp.loc[grp.index]]["perm_imp"].mean()
            base_imp = grp[is_base.loc[grp.index]]["perm_imp"].mean()
            print(f"  {p:<22s} {k:>3d}  cp={cp_imp:+.5f}  base={base_imp:+.5f}  ratio={cp_imp/(base_imp+1e-9):.2f}")
