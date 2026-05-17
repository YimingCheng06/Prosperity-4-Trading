"""Pure-stdlib feature extraction for counterparty-flow ML inference.

Mirrors the features built in train_cp_alpha.py but uses ONLY stdlib (math).
This module is meant to be imported by trader.py at inference time.

Conventions
-----------
- 1 bar = 100 timestamp units.
- MARKS is the canonical ordered list of counterparties; one-hot uses this order.
- Block trades (buyer in {Mark 01, Mark 67} AND seller in {Mark 22, Mark 49})
  are filtered out for both training and feature-building (do not contribute to flow).
- Time-to-expiry T (years) for vouchers: pulled from a shared constant; we use
  T = max(1e-4, (DAYS_LEFT - day_progress) / 252) with DAYS_LEFT supplied by caller.
"""

import math

MARKS = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]
MARK_TO_IDX = {m: i for i, m in enumerate(MARKS)}

# Block-trade filter (both informed parties on opposite sides => private flow)
BLOCK_BUYERS = {"Mark 01", "Mark 67"}
BLOCK_SELLERS = {"Mark 22", "Mark 49"}

# Voucher symbols and strikes
VOUCHER_STRIKES = {
    "VEV_4000": 4000.0,
    "VEV_4500": 4500.0,
    "VEV_5000": 5000.0,
    "VEV_5100": 5100.0,
    "VEV_5200": 5200.0,
    "VEV_5300": 5300.0,
    "VEV_5400": 5400.0,
    "VEV_5500": 5500.0,
    "VEV_6000": 6000.0,
    "VEV_6500": 6500.0,
}

ROLLING_WINDOWS_BARS = (5, 20, 50)  # bars; 1 bar = 100 ts


def is_block_trade(buyer, seller):
    return buyer in BLOCK_BUYERS and seller in BLOCK_SELLERS


def feature_names(symbol):
    """Return the ordered list of feature names for a given symbol."""
    names = []
    for m in MARKS:
        names.append(f"buy_{m.replace(' ', '_')}")
    for m in MARKS:
        names.append(f"sell_{m.replace(' ', '_')}")
    names.append("log_qty")
    names.append("dir_sign")
    for w in ROLLING_WINDOWS_BARS:
        for m in MARKS:
            names.append(f"flow_{w}_{m.replace(' ', '_')}")
    if symbol in VOUCHER_STRIKES:
        names.append("moneyness")
    return names


def extract_trade_features(buyer, seller, quantity, price, mid, symbol,
                            recent_flows, spot=None, T=None):
    """Extract features for a SINGLE trade event.

    Parameters
    ----------
    buyer, seller : str   "Mark XX" labels
    quantity      : float positive
    price         : float trade price
    mid           : float current mid for the symbol (used for direction)
    symbol        : str   product/symbol
    recent_flows  : dict  {window_bars: {mark: signed_qty_sum}}, where signed
                          means +qty if mark was buyer, -qty if mark was seller,
                          aggregated over the trailing `window_bars` bars.
    spot          : float current VELVETFRUIT mid (required for vouchers)
    T             : float years to expiry (required for vouchers)

    Returns
    -------
    list[float] feature vector aligned with feature_names(symbol).
    """
    feats = []
    # buyer one-hot
    for m in MARKS:
        feats.append(1.0 if buyer == m else 0.0)
    # seller one-hot
    for m in MARKS:
        feats.append(1.0 if seller == m else 0.0)
    # log quantity
    q = max(float(quantity), 1.0)
    feats.append(math.log(q))
    # direction sign relative to mid (taker proxy: trade above mid => buyer-initiated)
    if mid is None or mid <= 0:
        dir_sign = 0.0
    else:
        if price > mid:
            dir_sign = 1.0
        elif price < mid:
            dir_sign = -1.0
        else:
            dir_sign = 0.0
    feats.append(dir_sign)
    # rolling counterparty flows
    for w in ROLLING_WINDOWS_BARS:
        wf = recent_flows.get(w, {}) if recent_flows else {}
        for m in MARKS:
            feats.append(float(wf.get(m, 0.0)))
    # voucher moneyness
    if symbol in VOUCHER_STRIKES:
        K = VOUCHER_STRIKES[symbol]
        if spot is None or spot <= 0 or T is None or T <= 0:
            feats.append(0.0)
        else:
            feats.append(math.log(K / spot) / math.sqrt(T))
    return feats


def standardize(features, means, stds):
    out = []
    for v, mu, sd in zip(features, means, stds):
        if sd is None or sd <= 0:
            out.append(0.0)
        else:
            out.append((v - mu) / sd)
    return out


def linear_predict(std_features, coefs, intercept):
    s = float(intercept)
    for x, c in zip(std_features, coefs):
        s += x * c
    return s


class FlowTracker:
    """Maintain rolling signed-quantity flow per counterparty.

    For each non-block trade, call update(timestamp, buyer, seller, qty).
    Then call snapshot(timestamp) -> {window_bars: {mark: signed_qty}}.
    """

    def __init__(self, windows_bars=ROLLING_WINDOWS_BARS, bar_size=100):
        self.windows = list(windows_bars)
        self.bar_size = bar_size
        # store deque-like list of (ts, mark, signed_qty); we expire by oldest
        self._events = []  # list of tuples (ts, mark, signed_qty)

    def update(self, ts, buyer, seller, qty):
        if is_block_trade(buyer, seller):
            return
        q = float(qty)
        # Buyer is +q for buyer mark, seller is -q for seller mark
        self._events.append((ts, buyer, +q))
        self._events.append((ts, seller, -q))
        # Trim aggressively to the longest window
        max_w_ts = max(self.windows) * self.bar_size
        cutoff = ts - max_w_ts
        # drop from the front while older than cutoff
        i = 0
        for i, (ets, _, _) in enumerate(self._events):
            if ets >= cutoff:
                break
        else:
            i = len(self._events)
        if i > 0:
            self._events = self._events[i:]

    def snapshot(self, ts):
        out = {}
        for w in self.windows:
            cutoff = ts - w * self.bar_size
            agg = {}
            for ets, mark, sq in self._events:
                if ets >= cutoff:
                    agg[mark] = agg.get(mark, 0.0) + sq
            out[w] = agg
        return out
