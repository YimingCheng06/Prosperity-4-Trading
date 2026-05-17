"""Conditional event study: does the simple rule "actor X did Y at time t" predict
future Δmid on day 3 alone (out-of-sample)?

For each (product, actor, role), compute:
  Δmid_+k = mid_{t+k} - mid_{t} for trades where actor matches role.
  - Day 1+2 (in-sample) mean Δmid_+50
  - Day 3 (out-of-sample) mean Δmid_+50
  - t-stat for day-3 alone
"""
import pandas as pd
import numpy as np
from pathlib import Path

DATA_DIR = Path("/Users/emmett/Documents/Prosperity Trading/ROUND_4")

prices = pd.concat([
    pd.read_csv(DATA_DIR / f"prices_round_4_day_{d}.csv", sep=";").assign(dayfile=d)
    for d in [1, 2, 3]
], ignore_index=True)
trades = pd.concat([
    pd.read_csv(DATA_DIR / f"trades_round_4_day_{d}.csv", sep=";").assign(dayfile=d)
    for d in [1, 2, 3]
], ignore_index=True)
prices["mid_price"] = pd.to_numeric(prices["mid_price"], errors="coerce")

# Build mid lookup
mid_idx = prices.set_index(["dayfile", "timestamp", "product"])["mid_price"]


def mid_at(d, t, sym):
    try:
        return float(mid_idx.loc[(d, t, sym)])
    except (KeyError, TypeError):
        return np.nan


# For each trade, compute Δmid_+5/+25/+50 from current trade's timestamp
horizons = [5, 25, 50]  # in ticks (×100 ts)
out_rows = []
for _, r in trades.iterrows():
    d = r["dayfile"]; ts = r["timestamp"]; sym = r["symbol"]
    m_now = mid_at(d, ts, sym)
    if np.isnan(m_now): continue
    row = {"dayfile": d, "timestamp": ts, "symbol": sym,
           "buyer": r["buyer"], "seller": r["seller"], "qty": r["quantity"]}
    for k in horizons:
        m_k = mid_at(d, ts + 100 * k, sym)
        row[f"d_{k}"] = m_k - m_now if not np.isnan(m_k) else np.nan
    out_rows.append(row)

ev = pd.DataFrame(out_rows)
print(f"Event count: {len(ev)}")

# For each (symbol, actor, role), compute mean Δmid by day-set
ACTORS = sorted(set(ev["buyer"].dropna()) | set(ev["seller"].dropna()))
products_main = ["HYDROGEL_PACK", "VELVETFRUIT_EXTRACT", "VEV_4500", "VEV_5000",
                 "VEV_5100", "VEV_5200", "VEV_5300", "VEV_5400", "VEV_5500"]

print("\n=== Δmid_+50 by (product, actor, role): in-sample(d1+2) vs out-sample(d3) ===")
print(f"{'product':<22s} {'actor':<10s} {'role':<7s} {'n_in':>6s} {'mu_in':>8s} {'n_out':>6s} {'mu_out':>8s} {'t_out':>6s}")

results = []
for sym in products_main:
    sub_sym = ev[ev["symbol"] == sym]
    for actor in ACTORS:
        for role in ["buyer", "seller"]:
            mask = (sub_sym[role] == actor)
            sub = sub_sym[mask]
            if len(sub) < 30: continue
            # split
            in_s = sub[sub["dayfile"].isin([1, 2])]
            out_s = sub[sub["dayfile"] == 3]
            if len(out_s) < 10: continue
            # signed by role: for buyer, +Δmid is favorable for follow (buy when actor buys);
            # for seller, +Δmid means mid rose after sell → fade by buying.
            # We'll just report raw Δmid and let user interpret.
            d50_in = in_s["d_50"].dropna()
            d50_out = out_s["d_50"].dropna()
            if len(d50_out) < 10: continue
            mu_in = d50_in.mean(); mu_out = d50_out.mean()
            sd_out = d50_out.std(ddof=1)
            t_out = mu_out * np.sqrt(len(d50_out)) / sd_out if sd_out > 0 else 0
            row = {
                "product": sym, "actor": actor, "role": role,
                "n_in": len(d50_in), "mu_in": mu_in,
                "n_out": len(d50_out), "mu_out": mu_out, "t_out": t_out,
            }
            results.append(row)

res = pd.DataFrame(results)
res = res.sort_values("t_out", key=lambda x: x.abs(), ascending=False)
# Print top |t_out|, with sign-consistent in/out
print(res.head(40).to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

print("\n=== SIGN-CONSISTENT signals only (mu_in & mu_out same sign, |t_out|>=2) ===")
sign_consistent = res[(np.sign(res["mu_in"]) == np.sign(res["mu_out"])) & (res["t_out"].abs() >= 2.0)]
print(sign_consistent.to_string(index=False, float_format=lambda x: f"{x:+.3f}"))

print("\n=== ALSO show Δmid_+5 for the sign-consistent ones (do they fire fast?) ===")
ev_d5 = ev.dropna(subset=["d_5"])
for _, r in sign_consistent.iterrows():
    sym, actor, role = r["product"], r["actor"], r["role"]
    sub = ev_d5[(ev_d5["symbol"] == sym) & (ev_d5[role] == actor)]
    in_s = sub[sub["dayfile"].isin([1, 2])]
    out_s = sub[sub["dayfile"] == 3]
    if len(out_s) < 10: continue
    print(f"  {sym:<22s} {actor:<10s} {role:<7s}  Δmid+5: in={in_s['d_5'].mean():+.3f} out={out_s['d_5'].mean():+.3f}")

res.to_csv("ml_event_study.csv", index=False)
print("\nWrote ml_event_study.csv")
