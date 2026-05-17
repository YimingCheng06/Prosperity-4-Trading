"""
Round 4 trader — adapted from a proven Round 3 layered-MM framework.

Core architecture (Round 3 inheritance):
  * 5-layer passive market-making at ±0.15σ ±0.30σ ±0.50σ ±0.80σ ±1.20σ
    around a session-calibrated fair value (front-weighted 30/25/20/15/10%).
  * Mean-reversion aggressive take when |deviation| ≥ 1σ.
  * Inventory-aware quoting (long_heavy / short_heavy / panic states).
  * Order-book imbalance filter blocks adverse aggressive crosses.
  * Hard-coded fair/std priors blended with intraday session statistics
    once per restart, gated to avoid blending pure noise.

Round 4 adaptations:
  * Constants re-fit on Round 4 data (3 days × 10000 ticks).
  * TTE = 4/252 (Round 3 was 5/252).
  * Counterparty-flow fair-value nudge — Mark 01/67 buying → bias fair up
    by a small fraction of σ; Mark 22/49 selling → bias fair down.
  * Voucher pricing uses Round 4 smile coefs.
"""

from typing import List, Dict, Tuple, Any
import math
import json

try:
    from datamodel import OrderDepth, TradingState, Order  # noqa: F401
except Exception:
    Order = None


# ============================================================
# Round 4 calibration constants (computed from 3 days of mid_price data)
# ============================================================
# IV smile: IV(m) = a*m^2 + b*m + c, where m = ln(K/S)/sqrt(T)
SMILE_COEF = (0.131776, 0.016920, 0.228592)
SMILE_RESID_STD = 0.020   # rough std of IV residual on Round 4 data

# Underlying & base products — overall mean / intraday std
VEV_FAIR = 5247.65
VEV_STD = 16.73
HYDROGEL_FAIR = 9994.65
HYDROGEL_STD = 34.06

ROUND_TTE_DAYS = 4   # Round 4 starts with TTE = 4 days; treat as constant during sim

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Per-strike entry-edge multiplier — bigger = more conservative entry
EDGE_MULT = {5000: 1.5, 5100: 1.5, 5200: 1.0, 5300: 1.0, 5400: 0.9, 5500: 0.7}

# Direct mean-reversion params for VEV_4500/5000-5200 (where smile is unreliable)
# Values: (overall mean, intraday std) from Round 4 EDA
VEV_PROXY_PARAMS = {
    4500: (747.66, 16.75),
    5000: (251.14, 15.86),
    5100: (160.86, 14.16),
    5200: ( 88.99, 10.65),
}

POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
}
for K in STRIKES:
    POSITION_LIMITS[f"VEV_{K}"] = 300


# Counterparty roles (Round 4 — verified stable across 3 days)
AGG_BUYERS = {"Mark 01", "Mark 67"}
AGG_SELLERS = {"Mark 22", "Mark 49"}

# Fair-value nudge controls
CP_NUDGE_WINDOW = 30 * 100   # raw timestamp units (30 bars)
CP_NUDGE_FULL = 0.30         # max bias as fraction of σ when CP flow is "full strength"
CP_NUDGE_SAT = 25.0          # |net qty| at which the nudge saturates


# ============================================================
# Math helpers (stdlib only)
# ============================================================
def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def bs_call(S: float, K: float, T: float, sigma: float) -> float:
    if T <= 0 or sigma <= 0:
        return max(S - K, 0.0)
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    d2 = d1 - sigma * sqrtT
    return S * _norm_cdf(d1) - K * _norm_cdf(d2)


def implied_vol(price: float, S: float, K: float, T: float):
    intrinsic = max(S - K, 0.0)
    if price <= intrinsic + 1e-6:
        return None
    lo, hi = 1e-4, 5.0
    f_lo = bs_call(S, K, T, lo) - price
    f_hi = bs_call(S, K, T, hi) - price
    if f_lo * f_hi > 0:
        return None
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        f_m = bs_call(S, K, T, mid) - price
        if abs(f_m) < 1e-5 or (hi - lo) < 1e-6:
            return mid
        if f_lo * f_m < 0:
            hi = mid; f_hi = f_m
        else:
            lo = mid; f_lo = f_m
    return 0.5 * (lo + hi)


def smile_iv(m: float) -> float:
    a, b, c = SMILE_COEF
    return a * m * m + b * m + c


# ============================================================
# Trader
# ============================================================
class Trader:
    SESSION_FREEZE_N = 2000
    SESSION_RAMP_N = 100
    SESSION_MAX_W = 0.5
    SESSION_MAX_W_STD = 0.5
    SESSION_GATE = 0.3

    def _load_session(self, trader_data):
        if not trader_data:
            return {}
        try:
            d = json.loads(trader_data)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _save_session(self, sess):
        try:
            return json.dumps(sess, separators=(",", ":"))
        except Exception:
            return ""

    def _blend_fair(self, sess: dict, product: str, mid, fair_static: float, std_static: float):
        ps = sess.setdefault(product, {"sum": 0.0, "ssq": 0.0, "n": 0,
                                        "frozen_m": None, "frozen_s": None})
        if ps.get("frozen_m") is None and mid is not None:
            ps["sum"] += mid
            ps["ssq"] += mid * mid
            ps["n"] += 1
            if ps["n"] >= self.SESSION_FREEZE_N:
                m = ps["sum"] / ps["n"]
                v = max(0.0, ps["ssq"] / ps["n"] - m * m)
                ps["frozen_m"] = m
                ps["frozen_s"] = v ** 0.5
        if ps.get("frozen_m") is not None:
            sess_mean = ps["frozen_m"]
            sess_std = ps["frozen_s"]
            w_base = self.SESSION_MAX_W
        elif ps["n"] >= 50:
            m = ps["sum"] / ps["n"]
            v = max(0.0, ps["ssq"] / ps["n"] - m * m)
            sess_mean = m
            sess_std = v ** 0.5
            w_base = self.SESSION_MAX_W * min(1.0, ps["n"] / self.SESSION_RAMP_N)
        else:
            return fair_static, std_static
        offset = abs(sess_mean - fair_static)
        gate_thresh = self.SESSION_GATE * std_static
        if offset < gate_thresh:
            w_mean = 0.0
        else:
            w_mean = min(w_base, w_base * (offset - gate_thresh) / gate_thresh)
        w_std = min(1.0, ps["n"] / self.SESSION_RAMP_N) * self.SESSION_MAX_W_STD
        fair_eff = (1 - w_mean) * fair_static + w_mean * sess_mean
        std_eff = max(0.3 * std_static, (1 - w_std) * std_static + w_std * sess_std)
        return fair_eff, std_eff

    def _update_cp_flow(self, sess, state):
        flow = sess.setdefault("cpflow", {})
        ts_now = state.timestamp
        cutoff = ts_now - CP_NUDGE_WINDOW
        market_trades = getattr(state, "market_trades", {}) or {}
        for sym, trades in market_trades.items():
            bucket = flow.setdefault(sym, [])
            for t in trades:
                buyer = getattr(t, "buyer", None)
                seller = getattr(t, "seller", None)
                qty = int(getattr(t, "quantity", 0) or 0)
                net = 0
                if buyer in AGG_BUYERS:
                    net += qty
                if seller in AGG_SELLERS:
                    net -= qty
                if net != 0:
                    bucket.append([int(t.timestamp), net])
            flow[sym] = [x for x in bucket if x[0] >= cutoff][-200:]

    def _cp_nudge(self, sess, symbol, std):
        bucket = sess.get("cpflow", {}).get(symbol, [])
        if not bucket:
            return 0.0
        net = sum(x[1] for x in bucket)
        sat = max(-1.0, min(1.0, net / CP_NUDGE_SAT))
        return sat * CP_NUDGE_FULL * std

    def run(self, state):
        result: Dict[str, List] = {}
        sess = self._load_session(getattr(state, "traderData", "") or "")
        self._update_cp_flow(sess, state)

        vev_mid = self._get_mid(state.order_depths.get("VELVETFRUIT_EXTRACT"))

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            mid = self._get_mid(order_depth)

            if product == "HYDROGEL_PACK":
                f, s = self._blend_fair(sess, product, mid, HYDROGEL_FAIR, HYDROGEL_STD)
                f += self._cp_nudge(sess, product, s)
                orders = self._trade_mr(order_depth, position, product, f, s,
                                        POSITION_LIMITS[product])
            elif product == "VELVETFRUIT_EXTRACT":
                f, s = self._blend_fair(sess, product, mid, VEV_FAIR, VEV_STD)
                f += self._cp_nudge(sess, product, s)
                orders = self._trade_mr(order_depth, position, product, f, s,
                                        POSITION_LIMITS[product])
            elif product == "VEV_4000":
                f, s = self._blend_fair(sess, product, mid, VEV_FAIR - 4000, VEV_STD)
                f += self._cp_nudge(sess, product, s)
                orders = self._trade_mr(order_depth, position, product, f, s,
                                        POSITION_LIMITS[product])
            elif product.startswith("VEV_") and int(product.split("_")[1]) in VEV_PROXY_PARAMS:
                K = int(product.split("_")[1])
                fK, sK = VEV_PROXY_PARAMS[K]
                f, s = self._blend_fair(sess, product, mid, fK, sK)
                f += self._cp_nudge(sess, product, s)
                orders = self._trade_mr(order_depth, position, product, f, s,
                                        POSITION_LIMITS[product])
            elif product.startswith("VEV_"):
                K = int(product.split("_")[1])
                orders = self._trade_voucher(order_depth, position, product, K, vev_mid, sess)
            else:
                orders = []

            result[product] = orders

        return result, 0, self._save_session(sess)

    def _get_mid(self, order_depth):
        if order_depth is None:
            return None
        if not order_depth.sell_orders or not order_depth.buy_orders:
            return None
        ba = min(order_depth.sell_orders.keys())
        bb = max(order_depth.buy_orders.keys())
        bs = abs(order_depth.buy_orders[bb])
        as_ = abs(order_depth.sell_orders[ba])
        if bs + as_ == 0:
            return (ba + bb) / 2.0
        return (bb * as_ + ba * bs) / (bs + as_)

    def _trade_mr(self, order_depth, position: int, product: str,
                  fair: float, std: float, limit: int) -> List:
        orders: List = []
        if order_depth is None or Order is None:
            return orders

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is None and best_bid is None:
            return orders

        if best_bid is not None and best_ask is not None:
            bs = abs(order_depth.buy_orders[best_bid])
            as_ = abs(order_depth.sell_orders[best_ask])
            mid = (best_bid * as_ + best_ask * bs) / (bs + as_) if (bs + as_) > 0 else (best_bid + best_ask) / 2
        elif best_bid is not None:
            mid = best_bid + max(1, int(std * 2.00))
        else:
            mid = best_ask - max(1, int(std * 0.1))

        deviation = mid - fair

        bv1 = abs(order_depth.buy_orders[best_bid]) if best_bid is not None else 0
        av1 = abs(order_depth.sell_orders[best_ask]) if best_ask is not None else 0
        I_top = (bv1 - av1) / (bv1 + av1) if (bv1 + av1) > 0 else 0.0

        buy_room = limit - position
        sell_room = limit + position

        L1 = max(1, int(round(std * 0.15)))
        L2 = max(L1 + 1, int(round(std * 0.30)))
        L3 = max(L2 + 1, int(round(std * 0.50)))
        L4 = max(L3 + 1, int(round(std * 0.80)))
        L5 = max(L4 + 1, int(round(std * 1.20)))
        TAKE_T = max(2, int(round(std * 1.00)))
        REV_T = max(3, int(round(std * 2.25)))

        long_heavy = position > int(limit * 0.55)
        short_heavy = position < -int(limit * 0.55)
        panic_long = position > int(limit * 0.70)
        panic_short = position < -int(limit * 0.70)

        take_buy_max = fair - TAKE_T
        take_sell_min = fair + TAKE_T
        if long_heavy or panic_long:
            take_buy_max = fair - max(TAKE_T, L4)
            take_sell_min = fair - L1
        elif short_heavy or panic_short:
            take_buy_max = fair + L1
            take_sell_min = fair + max(TAKE_T, L4)

        buy_done = sell_done = 0
        allow_buy_take = (I_top >= -0.5) or short_heavy or panic_short
        allow_sell_take = (I_top <= 0.5) or long_heavy or panic_long
        if allow_buy_take:
            for price in sorted(order_depth.sell_orders.keys()):
                if price <= take_buy_max and buy_room - buy_done > 0:
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, buy_room - buy_done)
                    if qty > 0:
                        orders.append(Order(product, int(price), qty))
                        buy_done += qty
        if allow_sell_take:
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if price >= take_sell_min and sell_room - sell_done > 0:
                    vol = abs(order_depth.buy_orders[price])
                    qty = min(vol, sell_room - sell_done)
                    if qty > 0:
                        orders.append(Order(product, int(price), -qty))
                        sell_done += qty

        rev_cap = min(int(limit * 0.40), int(abs(deviation) / max(std, 1) * limit * 0.30))
        if deviation >= REV_T and position > -rev_cap:
            rev_room = min(rev_cap + position, sell_room - sell_done)
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if price >= fair and rev_room > 0:
                    vol = abs(order_depth.buy_orders[price])
                    qty = min(vol, rev_room)
                    if qty > 0:
                        orders.append(Order(product, int(price), -qty))
                        sell_done += qty
                        rev_room -= qty
        elif deviation <= -REV_T and position < rev_cap:
            rev_room = min(rev_cap - position, buy_room - buy_done)
            for price in sorted(order_depth.sell_orders.keys()):
                if price <= fair and rev_room > 0:
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, rev_room)
                    if qty > 0:
                        orders.append(Order(product, int(price), qty))
                        buy_done += qty
                        rev_room -= qty

        f = int(round(fair))
        per_side = max(40, int(limit * 0.50))
        layer_qtys = [int(per_side * w) for w in [0.3, 0.25, 0.2, 0.15, 0.1]]

        bid_layers = list(zip([f - L1, f - L2, f - L3, f - L4, f - L5], layer_qtys))
        ask_layers = list(zip([f + L1, f + L2, f + L3, f + L4, f + L5], layer_qtys))

        if deviation >= TAKE_T:
            ask_layers = list(zip([f, f + L1, f + L2, f + L3, f + L4], layer_qtys))
            bid_layers = list(zip([f - L2, f - L3, f - L4, f - L5], layer_qtys[:4]))
        elif deviation <= -TAKE_T:
            bid_layers = list(zip([f, f - L1, f - L2, f - L3, f - L4], layer_qtys))
            ask_layers = list(zip([f + L2, f + L3, f + L4, f + L5], layer_qtys[:4]))

        if panic_long:
            ask_layers = [(f, per_side // 2), (f + L1, per_side // 2)]
            bid_layers = []
        elif long_heavy:
            ask_layers = [(f + L1, per_side // 4), (f + L2, per_side // 4),
                          (f + L3, per_side // 4), (f + L4, per_side // 4),
                          (f + L5, per_side // 4)]
            bid_layers = [(f - L4, per_side // 4)]
        elif panic_short:
            bid_layers = [(f, per_side // 2), (f - L1, per_side // 2)]
            ask_layers = []
        elif short_heavy:
            bid_layers = [(f - L1, per_side // 4), (f - L2, per_side // 4),
                          (f - L3, per_side // 4), (f - L4, per_side // 4),
                          (f - L5, per_side // 4)]
            ask_layers = [(f + L4, per_side // 4)]

        rb = buy_room - buy_done
        rs = sell_room - sell_done
        max_bid = (best_ask - 1) if best_ask is not None else 10**9
        min_ask = (best_bid + 1) if best_bid is not None else 1
        for price, qty in bid_layers:
            if rb <= 0:
                break
            q = min(qty, rb)
            if q > 0 and 1 <= price <= max_bid:
                orders.append(Order(product, int(price), q))
                rb -= q
        for price, qty in ask_layers:
            if rs <= 0:
                break
            q = min(qty, rs)
            if q > 0 and price >= min_ask:
                orders.append(Order(product, int(price), -q))
                rs -= q

        return orders

    def _trade_voucher(self, order_depth, position: int, product: str,
                       K: int, S, sess) -> List:
        orders: List = []
        if order_depth is None or S is None or Order is None:
            return orders
        if not order_depth.sell_orders and not order_depth.buy_orders:
            return orders

        T = ROUND_TTE_DAYS / 252.0
        if T <= 0:
            return orders

        limit = POSITION_LIMITS[product]
        intrinsic = max(S - K, 0)

        if K <= 4000:
            return self._floor_intrinsic_mm(order_depth, position, product, S, K, limit)
        if K >= 6000:
            return self._unwind_only(order_depth, position, product)

        m = math.log(K / S) / math.sqrt(T)
        fair_iv = smile_iv(m)
        fair_price = bs_call(S, K, T, fair_iv)

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        mkt_mid = (best_ask + best_bid) / 2 if (best_ask and best_bid) else (best_ask or best_bid)
        if mkt_mid is None:
            return orders

        cur_iv = implied_vol(mkt_mid, S, K, T) if mkt_mid > intrinsic + 0.5 else None
        edge_threshold = SMILE_RESID_STD * EDGE_MULT.get(K, 1.0)

        buy_room = limit - position
        sell_room = limit + position
        buy_done = sell_done = 0

        heavy_long = position > int(limit * 0.50)
        heavy_short = position < -int(limit * 0.50)
        panic_long = position > int(limit * 0.70)
        panic_short = position < -int(limit * 0.70)

        EAT_EDGE = max(0.5, fair_price * 0.005)
        if (best_ask is not None and best_ask <= fair_price - EAT_EDGE
                and buy_room > 0 and not (heavy_long or panic_long)):
            for p in sorted(order_depth.sell_orders.keys()):
                if p <= fair_price - EAT_EDGE and buy_room - buy_done > 0:
                    vol = abs(order_depth.sell_orders[p])
                    qty = min(vol, buy_room - buy_done)
                    orders.append(Order(product, int(p), qty))
                    buy_done += qty
        if (best_bid is not None and best_bid >= fair_price + EAT_EDGE
                and sell_room > 0 and not (heavy_short or panic_short)):
            for p in sorted(order_depth.buy_orders.keys(), reverse=True):
                if p >= fair_price + EAT_EDGE and sell_room - sell_done > 0:
                    vol = abs(order_depth.buy_orders[p])
                    qty = min(vol, sell_room - sell_done)
                    orders.append(Order(product, int(p), -qty))
                    sell_done += qty

        if cur_iv is not None and not (panic_long or panic_short):
            iv_diff = cur_iv - fair_iv
            if iv_diff > edge_threshold and sell_room - sell_done > 0 and not heavy_short:
                px = max(int(math.ceil(fair_price + EAT_EDGE)),
                         best_bid + 1 if best_bid else int(mkt_mid))
                qty = min(30, sell_room - sell_done)
                if qty > 0:
                    orders.append(Order(product, px, -qty))
                    sell_done += qty
            elif iv_diff < -edge_threshold and buy_room - buy_done > 0 and not heavy_long:
                px = min(int(math.floor(fair_price - EAT_EDGE)),
                         best_ask - 1 if best_ask else int(mkt_mid))
                qty = min(30, buy_room - buy_done)
                if qty > 0:
                    orders.append(Order(product, px, qty))
                    buy_done += qty

        f_int = int(round(fair_price))
        if f_int >= 1:
            mm_qty = max(5, min(30, limit // 10))
            mm_bid = f_int - 1
            mm_ask = f_int + 1
            if best_bid is not None:
                mm_bid = min(mm_bid, best_bid)
            if best_ask is not None:
                mm_ask = max(mm_ask, best_ask)
            allow_bid = not (heavy_long or panic_long)
            allow_ask = not (heavy_short or panic_short)
            if allow_bid and buy_room - buy_done > 0 and mm_bid >= 1:
                orders.append(Order(product, mm_bid, min(mm_qty, buy_room - buy_done)))
            if allow_ask and sell_room - sell_done > 0:
                orders.append(Order(product, mm_ask, -min(mm_qty, sell_room - sell_done)))

        if panic_long and order_depth.buy_orders:
            bb = max(order_depth.buy_orders.keys())
            if bb >= 1:
                qty = min(30, position, abs(order_depth.buy_orders[bb]))
                if qty > 0:
                    orders.append(Order(product, int(bb), -qty))
        if panic_short and order_depth.sell_orders:
            ba = min(order_depth.sell_orders.keys())
            qty = min(30, abs(position), abs(order_depth.sell_orders[ba]))
            if qty > 0:
                orders.append(Order(product, int(ba), qty))

        return orders

    def _unwind_only(self, order_depth, position, product) -> List:
        orders: List = []
        if Order is None or position == 0 or order_depth is None:
            return orders
        if position > 0 and order_depth.buy_orders:
            best_bid = max(order_depth.buy_orders.keys())
            if best_bid >= 1:
                vol = abs(order_depth.buy_orders[best_bid])
                qty = min(vol, position)
                if qty > 0:
                    orders.append(Order(product, int(best_bid), -qty))
        elif position < 0 and order_depth.sell_orders:
            best_ask = min(order_depth.sell_orders.keys())
            vol = abs(order_depth.sell_orders[best_ask])
            qty = min(vol, abs(position))
            if qty > 0:
                orders.append(Order(product, int(best_ask), qty))
        return orders

    def _floor_intrinsic_mm(self, order_depth, position, product, S, K, limit) -> List:
        orders: List = []
        if Order is None:
            return orders
        intrinsic = max(S - K, 0)
        buy_room = limit - position
        sell_room = limit + position
        fair = intrinsic
        if order_depth.sell_orders:
            for p in sorted(order_depth.sell_orders.keys()):
                if p <= fair - 0.5 and buy_room > 0:
                    vol = abs(order_depth.sell_orders[p])
                    qty = min(vol, buy_room)
                    orders.append(Order(product, int(p), qty))
                    buy_room -= qty
        if order_depth.buy_orders:
            for p in sorted(order_depth.buy_orders.keys(), reverse=True):
                if p >= fair + 0.5 and sell_room > 0:
                    vol = abs(order_depth.buy_orders[p])
                    qty = min(vol, sell_room)
                    orders.append(Order(product, int(p), -qty))
                    sell_room -= qty
        return orders
