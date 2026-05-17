from datamodel import OrderDepth, TradingState, Order
from typing import List, Dict, Tuple
import math
import json


# ============================================================
# INLINED ML overlay (was cp_features.py + cp_alpha.json)
# Single-file submission requirement — everything self-contained below.
# Source: trained on Round 4 days 1-2, validated on day 3:
#   VELVETFRUIT_EXTRACT h=50 → IC_test=+0.155, Sharpe_test=+3.19, n_test=407.
# ============================================================
_MARKS = ["Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67"]
_BLOCK_BUYERS = {"Mark 01", "Mark 67"}
_BLOCK_SELLERS = {"Mark 22", "Mark 49"}
_ROLLING_WINDOWS_BARS = (5, 20, 50)
_BAR_SIZE = 100

# Ridge weights for VELVETFRUIT_EXTRACT, horizon=50 bars (the only reliable model).
# Feature order matches _build_velv_features() below.
_ML_INTERCEPT = 7.364139880821253e-06
_ML_COEFS = [0.00016019813078192288, 0.0001345492937022992, 1.921250198560503e-05, 0.0, -2.6288485463016686e-05, -0.00024642638769069455, 0.0, -4.5950953153696153e-05, 8.156578601302665e-05, 0.00013192160088571408, 0.0, 1.3378792301487249e-05, -7.737915421096932e-05, 0.0, 0.00012245520136623244, 0.00036183578781960883, 2.233319047972629e-06, 2.260623636639482e-06, 1.6113075219293288e-05, 4.449122607390719e-05, -2.699897504493103e-05, -3.851028642575248e-05, 0.0, 1.2195082370077445e-05, -3.063493420671937e-05, -8.319518224892039e-05, 3.731107418193864e-05, -8.358806336348895e-05, 4.310000566023721e-05, 0.0, 3.0792508791292686e-05, -7.081779032662689e-05, 3.055855203040874e-05, -0.00013106360944137554, 7.996434803468866e-05, 0.0001115881531297926, 0.0]
_ML_MEANS = [0.20745341614906831, 0.253416149068323, 0.02360248447204969, 0.0, 0.014906832298136646, 0.5006211180124224, 0.0, 0.21490683229813665, 0.26583850931677017, 0.02360248447204969, 0.0, 0.012422360248447204, 0.4832298136645963, 0.0, 1.6653325247754025, 0.016149068322981366, -0.03229813664596273, -0.10683229813664596, -0.04472049689440994, 0.13291925465838508, -0.004968944099378882, 0.055900621118012424, 0.0, -0.0782608695652174, 0.058385093167701865, -0.2124223602484472, 0.13291925465838508, -0.02111801242236025, 0.12049689440993788, 0.0, 0.09565217391304348, 0.8, -0.4260869565217391, -0.09192546583850932, -0.024844720496894408, -0.3527950310559006, 0.0]
_ML_STDS = [0.40548304067759866, 0.4349671303210219, 0.15180713816812524, 1.0, 0.12118010830566218, 0.4999996142122665, 1.0, 0.41075769710343385, 0.44177867341125243, 0.15180713816812527, 1.0, 0.11076120807532432, 0.49971868171029676, 1.0, 0.3293742331066508, 0.9998695952934518, 1.632420086768703, 2.6774448796711545, 0.9757240102807093, 1.9409857391137524, 0.6612430644928335, 2.4389399163549754, 1.0, 3.4612440884776694, 5.487677432507878, 1.6890865539779878, 3.855266244159528, 1.2171859378053587, 5.204788979826596, 1.0, 5.594766274638882, 8.818444814370567, 2.6268683794818797, 5.790246405416484, 1.8244383482176825, 8.826682178188921, 1.0]

# Voucher indirect ML bias scale; start dormant (0.0).
ML_VOUCHER_BIAS_SCALE = 0.0


def _is_block_trade(buyer, seller):
    return buyer in _BLOCK_BUYERS and seller in _BLOCK_SELLERS


def _build_velv_features(buyer, seller, quantity, price, mid, recent_flows):
    """Pure-stdlib feature vector for the VELVETFRUIT_EXTRACT Ridge model.
    Order: 7 buyer one-hot, 7 seller one-hot, log_qty, dir_sign,
           7 flow_5, 7 flow_20, 7 flow_50.   (37 features for VELVETFRUIT, no moneyness)
    """
    feats = []
    for m in _MARKS:
        feats.append(1.0 if buyer == m else 0.0)
    for m in _MARKS:
        feats.append(1.0 if seller == m else 0.0)
    q = max(float(quantity), 1.0)
    feats.append(math.log(q))
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
    for w in _ROLLING_WINDOWS_BARS:
        wf = recent_flows.get(w, {}) if recent_flows else {}
        for m in _MARKS:
            feats.append(float(wf.get(m, 0.0)))
    return feats


def _ml_velvetfruit_pred(flow_snapshot, market_trade, mid):
    """Predict VELVETFRUIT forward log-return at h=50. Returns 0.0 on any failure."""
    if market_trade is None or mid is None:
        return 0.0
    try:
        feats = _build_velv_features(
            buyer=getattr(market_trade, "buyer", None),
            seller=getattr(market_trade, "seller", None),
            quantity=int(getattr(market_trade, "quantity", 0) or 0),
            price=float(getattr(market_trade, "price", 0) or 0),
            mid=float(mid),
            recent_flows=flow_snapshot,
        )
        s = float(_ML_INTERCEPT)
        for v, mu, sd, c in zip(feats, _ML_MEANS, _ML_STDS, _ML_COEFS):
            if sd is None or sd <= 0:
                continue
            s += ((v - mu) / sd) * c
        return s
    except Exception:
        return 0.0


class _FlowTracker:
    """Maintain rolling signed-quantity flow per counterparty (block trades excluded)."""
    def __init__(self):
        self._events = []   # list of (ts, mark, signed_qty)

    def update(self, ts, buyer, seller, qty):
        if _is_block_trade(buyer, seller):
            return
        q = float(qty)
        self._events.append((ts, buyer, +q))
        self._events.append((ts, seller, -q))
        max_w_ts = max(_ROLLING_WINDOWS_BARS) * _BAR_SIZE
        cutoff = ts - max_w_ts
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
        for w in _ROLLING_WINDOWS_BARS:
            cutoff = ts - w * _BAR_SIZE
            agg = {}
            for ets, mark, sq in self._events:
                if ets >= cutoff:
                    agg[mark] = agg.get(mark, 0.0) + sq
            out[w] = agg
        return out


# ============================================================
# Calibration constants — 用 colab_factor_mining.ipynb 训练后替换
# ============================================================
# IV smile: IV(m) = a*m^2 + b*m + c, where m = ln(K/S)/sqrt(T)
# Cleaned calibration: K=4500/4000/6000/6500 排除，Huber 鲁棒回归，per-day demean
# EXACT Round 3 constants — empirically the best-performing baseline (52k live).
# Round 4 EDA fits hurt PnL by ~2-4k; reverted.
SMILE_COEF = (0.030889, 0.004210, 0.192393)
SMILE_RESID_STD = 0.007357

VEV_FAIR = 5255.4
VEV_STD = 15.16
HYDROGEL_FAIR = 9989.4
HYDROGEL_STD = 31.92

# TTE: Round 4 spec says 4 days, but Round 3's 5 paired better with R3 smile coefs.
# Keep at 4 for theoretical correctness; the smile-residual logic only uses
# fair/EAT_EDGE which are spot-driven, not strongly TTE-sensitive.
ROUND_TTE_DAYS = 4

STRIKES = [4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500]

# Per-strike IV 阈值倍率：来自 colab cell 8 mean-revert 强度
# K=5500 corr=-0.16 (极强) → 倍率小，更早入场
# K=5100/5000 corr≈-0.02 (弱) → 倍率大，避免噪声
EDGE_MULT = {5000: 1.5, 5100: 1.5, 5200: 1.0, 5300: 1.0, 5400: 0.9, 5500: 0.7}

# VEV vouchers 直接 MR 参数：(3-day mean, intra-day std)
# diff_cor 与 VEV 0.71-0.77，每日 mean 稳定（vs VEV 漂移大）
# 5300/5400/5500 跨日 mean monotonic 衰减（theta decay）→ 不适合 hardcoded
VEV_PROXY_PARAMS = {
    4500: (750.0, 17.3),
    5000: (255.0, 16.1),
    5100: (167.0, 13.8),
    5200: (95.0, 10.4),
    5300: (46.5, 6.9),
    5400: (16.0, 3.2),
    5500: (6.6, 1.5),
}

POSITION_LIMITS = {
    "HYDROGEL_PACK": 200,
    "VELVETFRUIT_EXTRACT": 200,
}
for K in STRIKES:
    POSITION_LIMITS[f"VEV_{K}"] = 300


# ============================================================
# Stuck-inventory force-unwind tuning knobs
# When position pins at heavy for many consecutive ticks, our existing
# panic_long/panic_short logic may never fill (asks at fair-L1 don't cross
# the market bid). Force-unwind aggressively crosses the spread to free
# capacity for the rest of the sim.
# ============================================================
STUCK_LIMIT_BARS = 99999       # DISABLED — backtest shows force-unwind breaks MR strategy
                               # regardless of threshold. Re-enable with surgical fix only.
STUCK_THRESHOLD_FRAC = 0.99    # effectively disabled
ESCAPE_THRESHOLD_FRAC = 0.40   # stop force-unwinding once below this
FORCE_UNWIND_QTY_PER_TICK = 30 # max qty to liquidate per tick during force-unwind


# ============================================================
# Black-Scholes (纯 math，避免 scipy 依赖)
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


def implied_vol(price: float, S: float, K: float, T: float) -> float:
    """Brent 风格的二分搜索，返回 IV。无解时返回 None。"""
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


def _bs_delta(S: float, K: float, T: float, sigma: float) -> float:
    """Black-Scholes call delta = N(d1)."""
    if T <= 0 or sigma <= 0 or S <= 0:
        return 1.0 if S > K else 0.0
    sqrtT = math.sqrt(T)
    d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
    return _norm_cdf(d1)


def _portfolio_delta(positions: dict, S: float, T: float) -> float:
    """Sum of pos_i * delta_i for all voucher positions. Returns total delta in
    units of VELVETFRUIT shares (1 voucher delta = 1 VELVETFRUIT equivalent)."""
    if S is None or S <= 0 or T <= 0:
        return 0.0
    sqrtT = math.sqrt(T)
    total = 0.0
    for K in STRIKES:
        sym = f"VEV_{K}"
        pos = positions.get(sym, 0)
        if pos == 0:
            continue
        try:
            m = math.log(K / S) / sqrtT
            sigma = max(0.001, smile_iv(m))
            delta = _bs_delta(S, K, T, sigma)
        except Exception:
            delta = 1.0 if S > K else 0.0
        total += pos * delta
    return total


# Delta-hedge parameters
HEDGE_DELTA_GAIN = 1.0  # 1.0 = full delta hedge (perfectly offset). Lower = partial hedge.
                         # Tune in 0.5-1.5 range based on submission feedback.


# ============================================================
# Trader 主类
# ============================================================
class Trader:

    # 一次性会话校准：前 N 个 tick 累积 mid，>=N 后冻结 mean 永不更新
    # 避免 EWMA 陷阱：不跟踪 intraday drift，只校准 regime level
    # 阈值化：仅当 |session_mean - hardcoded| 超过 SESSION_GATE σ 时才 blend
    # backtest 偏差小（hardcoded 来自训练日）→ 几乎不 blend；live 偏差 0.5-0.7σ → 满 w blend
    SESSION_FREEZE_N = 2000
    SESSION_RAMP_N = 100       # 1000-tick live session needs fast engagement
    SESSION_MAX_W = 0.5
    SESSION_MAX_W_STD = 0.5    # std blend 强度（独立于 mean blend）
    SESSION_GATE = 0.3   # in units of std_static  (live drifts 0.3-0.7σ; 0.6 missed most)

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

    def _get_flow_tracker(self, sess):
        """Lazy-initialize FlowTracker, hydrating events from sess."""
        raw_events = sess.get("_flow_events", [])
        ft = _FlowTracker()
        try:
            ft._events = [tuple(e) for e in raw_events]
        except Exception:
            ft._events = []
        return ft

    def _save_flow_tracker(self, sess, ft):
        if ft is None:
            return
        try:
            sess["_flow_events"] = [list(e) for e in ft._events[-2000:]]
        except Exception:
            sess["_flow_events"] = []

    def _track_stuck(self, sess, product: str, position: int, limit: int) -> int:
        """Per-symbol stuck-inventory tracker.

        Returns:
            +1 if force-sell (long stuck) should activate this tick,
            -1 if force-buy  (short stuck) should activate this tick,
             0 otherwise.

        State persisted in sess["_stuck"][product] = {"ticks": int, "active": int}.
        active=+1/-1 latches once triggered and only releases when |pos|/limit
        drops below ESCAPE_THRESHOLD_FRAC (hysteresis to prevent oscillation).
        """
        if limit <= 0:
            return 0
        stuck_map = sess.setdefault("_stuck", {})
        st = stuck_map.setdefault(product, {"ticks": 0, "active": 0})
        try:
            ticks = int(st.get("ticks", 0))
            active = int(st.get("active", 0))
        except Exception:
            ticks = 0
            active = 0

        heavy_thresh = int(limit * STUCK_THRESHOLD_FRAC)
        escape_thresh = int(limit * ESCAPE_THRESHOLD_FRAC)

        # Update heavy counter
        if position >= heavy_thresh:
            # If already actively force-unwinding short side, reset.
            if active == -1:
                active = 0
                ticks = 0
            ticks += 1 if (active == 0 or active == +1) else 1
            side = +1
        elif position <= -heavy_thresh:
            if active == +1:
                active = 0
                ticks = 0
            ticks += 1
            side = -1
        else:
            ticks = 0
            side = 0

        # Trigger force-unwind once threshold crossed.
        if active == 0 and side != 0 and ticks >= STUCK_LIMIT_BARS:
            active = side

        # Release once position drops back inside escape band.
        if active == +1 and position < escape_thresh:
            active = 0
            ticks = 0
        elif active == -1 and position > -escape_thresh:
            active = 0
            ticks = 0

        st["ticks"] = ticks
        st["active"] = active
        return active

    def _blend_fair(self, sess: dict, product: str, mid, fair_static: float, std_static: float):
        """同时校准 mean 和 std。返回 (fair_eff, std_eff)。"""
        ps = sess.setdefault(product, {"sum": 0.0, "ssq": 0.0, "n": 0,
                                        "frozen_m": None, "frozen_s": None})
        # accumulate until freeze
        if ps.get("frozen_m") is None and mid is not None:
            ps["sum"] += mid
            ps["ssq"] += mid * mid
            ps["n"] += 1
            if ps["n"] >= self.SESSION_FREEZE_N:
                m = ps["sum"] / ps["n"]
                v = max(0.0, ps["ssq"] / ps["n"] - m * m)
                ps["frozen_m"] = m
                ps["frozen_s"] = v ** 0.5
        # compute blend weight
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
        # gating: 仅当偏差 >= SESSION_GATE σ 时才 blend mean
        offset = abs(sess_mean - fair_static)
        gate_thresh = self.SESSION_GATE * std_static
        if offset < gate_thresh:
            w_mean = 0.0
        else:
            # ramp from 0 (at gate) to w_base (at 2*gate)
            w_mean = min(w_base, w_base * (offset - gate_thresh) / gate_thresh)
        # std 总是 blend，更激进（live 普遍 std=0.45×hardcoded，需要充分校准让 take/layer 不过宽）
        w_std = min(1.0, ps["n"] / self.SESSION_RAMP_N) * self.SESSION_MAX_W_STD
        fair_eff = (1 - w_mean) * fair_static + w_mean * sess_mean
        std_eff = max(0.3 * std_static, (1 - w_std) * std_static + w_std * sess_std)
        return fair_eff, std_eff

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}

        sess = self._load_session(getattr(state, "traderData", "") or "")

        vev_mid = self._get_mid(state.order_depths.get("VELVETFRUIT_EXTRACT"))

        # ML overlay: ingest market trades (all symbols) into FlowTracker, snapshot.
        ft = self._get_flow_tracker(sess)
        ts_now = int(getattr(state, "timestamp", 0) or 0)
        mt_all = getattr(state, "market_trades", {}) or {}
        last_velv_trade = None
        for sym, trs in mt_all.items():
            if trs is None:
                continue
            for tr in trs:
                if ft is not None:
                    try:
                        ft.update(int(getattr(tr, "timestamp", ts_now) or ts_now),
                                  getattr(tr, "buyer", None),
                                  getattr(tr, "seller", None),
                                  int(getattr(tr, "quantity", 0) or 0))
                    except Exception:
                        pass
                if sym == "VELVETFRUIT_EXTRACT":
                    last_velv_trade = tr
        flow_snap = ft.snapshot(ts_now) if ft is not None else {}
        self._save_flow_tracker(sess, ft)

        ml_pred_velv = _ml_velvetfruit_pred(flow_snap, last_velv_trade, vev_mid)

        # ============================================================
        # ADVERSE-FILL DEFENSE — own_trades-driven (route C revised)
        # ============================================================
        # New thesis: every prior CP integration (using market_trades signals)
        # hurt live PnL. Switch to a fundamentally different signal source:
        # `state.own_trades` — the trades WE got filled on. When an INFORMED
        # counterparty hits our quote, that's a 100% confirmed adverse-selection
        # event. We KNOW we got filled at the wrong side. React directly.
        #
        # Trigger:
        #   Mark 22/49 sold to us (we bought from them) → we're long, wrong side
        #     → EDA: their sells predict ↓ → bias fair DOWN → MR strategy sells
        #   Mark 01/67 bought from us (we sold to them) → we're short, wrong side
        #     → EDA: their buys predict ↑ → bias fair UP → MR strategy buys
        #
        # This signal is much sparser (only fires when we got hit by informed)
        # but much higher confidence — confirmed ex-post adverse selection.
        # Note: backtest can't observe this — counterparties in own_trades come
        # from the historical book aggregate, not specific Mark IDs. Live only.
        _MMS = ("Mark 14", "Mark 38", "Mark 55")
        _MARK_SET = ("Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67")
        _INF_BUYERS = ("Mark 01", "Mark 67")  # informed buyers (when they buy from us)
        _INF_SELLERS = ("Mark 22", "Mark 49") # informed sellers (when they sell to us)
        ADVERSE_BIAS_SIGMA = 1.5   # tuned up from 0.7 (which gave +1k live)
                                    # CONTRARIAN direction confirmed by 53k vs 52k vanilla.
                                    # Linear-scaling assumption: 1.5σ → +2-3k more.
        ADVERSE_BARS = 30           # decay window after adverse fill

        # Per-symbol contrarian-CP tracking (extended from VELVETFRUIT-only after 53k confirm)
        cp_adverse_all = sess.setdefault("_cp_adverse_all", {})  # {sym: {"buy_event_ts","sell_event_ts"}}

        own_trades_all = getattr(state, "own_trades", {}) or {}
        for sym, trades in own_trades_all.items():
            cp_adv = cp_adverse_all.setdefault(sym,
                                                {"buy_event_ts": None, "sell_event_ts": None})
            for tr in (trades or []):
                buyer = getattr(tr, "buyer", None)
                seller = getattr(tr, "seller", None)
                t_ts = int(getattr(tr, "timestamp", ts_now) or ts_now)
                cp = None
                we_bought = None
                if seller in _MARK_SET and buyer not in _MARK_SET:
                    cp = seller; we_bought = True
                elif buyer in _MARK_SET and seller not in _MARK_SET:
                    cp = buyer; we_bought = False
                if cp is None:
                    continue
                if we_bought and cp in _INF_SELLERS:
                    cp_adv["sell_event_ts"] = t_ts
                if (not we_bought) and cp in _INF_BUYERS:
                    cp_adv["buy_event_ts"] = t_ts

        def _cp_bias(std, sym="VELVETFRUIT_EXTRACT"):
            """CONTRARIAN per-symbol bias.
            Mark 22/49 sold to us in `sym` → keep long → bias `sym` fair UP.
            Mark 01/67 bought from us in `sym` → keep short → bias `sym` fair DOWN.
            Live-only signal (backtest own_trades has no Mark IDs).
            """
            cp_adv = cp_adverse_all.get(sym, {})
            bias = 0.0
            for ts_attr, sign in (("buy_event_ts", -1), ("sell_event_ts", +1)):
                ts_x = cp_adv.get(ts_attr)
                if ts_x is None:
                    continue
                gap_bars = (ts_now - ts_x) // 100
                if 0 <= gap_bars < ADVERSE_BARS:
                    weight = 1.0 - (gap_bars / ADVERSE_BARS)
                    bias += sign * ADVERSE_BIAS_SIGMA * weight * std
            cap = ADVERSE_BIAS_SIGMA * std
            return max(-cap, min(cap, bias))

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            mid = self._get_mid(order_depth)

            # Force-unwind ONLY enabled for base products (passive MM pin issue).
            # Vouchers use MR strategy where holding heavy is intentional — disabled.
            limit_p = POSITION_LIMITS.get(product, 0)
            if product in ("HYDROGEL_PACK", "VELVETFRUIT_EXTRACT") and limit_p > 0:
                force_unwind = self._track_stuck(sess, product, position, limit_p)
            else:
                force_unwind = 0

            if product == "HYDROGEL_PACK":
                fair_eff, std_eff = self._blend_fair(sess, product, mid, HYDROGEL_FAIR, HYDROGEL_STD)
                fair_eff += _cp_bias(std_eff, product)
                # HP warmup: skip trading entirely until session has 100+ obs, then
                # use observed mid as the fair anchor (live regime self-calibrated).
                # This avoids the catastrophic opening drawdown (-5k by tick 8k seen in
                # live PnL chart) caused by static prior 9989 mismatching live mid.
                hp_n_obs = sess.get(product, {}).get("n", 0)
                if hp_n_obs < 100:
                    orders = []   # WARMUP — no HP trades
                else:
                    orders = self._trade_mr(order_depth, position, product,
                                            fair=fair_eff, std=std_eff,
                                            limit=POSITION_LIMITS[product],
                                            force_unwind=force_unwind)
            elif product == "VELVETFRUIT_EXTRACT":
                fair_eff, std_eff = self._blend_fair(sess, product, mid, VEV_FAIR, VEV_STD)
                fair_eff += _cp_bias(std_eff, product)
                orders = self._trade_mr(order_depth, position, product,
                                        fair=fair_eff, std=std_eff,
                                        limit=POSITION_LIMITS[product],
                                        force_unwind=force_unwind)
            elif product == "VEV_4000":
                fair_eff, std_eff = self._blend_fair(sess, product, mid,
                                                    VEV_FAIR - 4000, VEV_STD)
                fair_eff += _cp_bias(std_eff, product)
                orders = self._trade_mr(order_depth, position, product,
                                        fair=fair_eff, std=std_eff,
                                        limit=POSITION_LIMITS[product],
                                        force_unwind=force_unwind)
            elif product.startswith("VEV_") and int(product.split("_")[1]) in VEV_PROXY_PARAMS:
                K = int(product.split("_")[1])
                fair_K, std_K = VEV_PROXY_PARAMS[K]
                fair_eff, std_eff = self._blend_fair(sess, product, mid, fair_K, std_K)
                fair_eff += _cp_bias(std_eff, product)
                orders = self._trade_mr(order_depth, position, product,
                                        fair=fair_eff, std=std_eff,
                                        limit=POSITION_LIMITS[product],
                                        force_unwind=force_unwind)
            elif product.startswith("VEV_"):
                K = int(product.split("_")[1])
                S_voucher = vev_mid
                if (S_voucher is not None and ML_VOUCHER_BIAS_SCALE != 0.0
                        and ml_pred_velv != 0.0):
                    S_voucher = S_voucher * (1.0 + ml_pred_velv * ML_VOUCHER_BIAS_SCALE)
                # Per-voucher std for CP bias (intra-day std from EDA)
                _voucher_std = {5300: 6.9, 5400: 3.2, 5500: 1.5,
                                6000: 0.5, 6500: 0.5}.get(K, 1.0)
                _cp_bias_voucher = _cp_bias(_voucher_std, product)
                orders = self._trade_voucher(order_depth, position, product, K, S_voucher,
                                             force_unwind=force_unwind,
                                             cp_bias=_cp_bias_voucher)
            else:
                orders = []

            result[product] = orders

        return result, 0, self._save_session(sess)

    def _portfolio_delta(self, state: TradingState, S, T) -> float:
        """∑(pos_i * Δ_i)，用 smile IV 算 BS delta。S 缺失或 T<=0 时返回 0。"""
        if S is None or T <= 0:
            return 0.0
        sqrtT = math.sqrt(T)
        total = 0.0
        for K in STRIKES:
            pos = state.position.get(f"VEV_{K}", 0)
            if pos == 0:
                continue
            try:
                m = math.log(K / S) / sqrtT
                sigma = smile_iv(m)
                d1 = (math.log(S / K) + 0.5 * sigma * sigma * T) / (sigma * sqrtT)
                delta = _norm_cdf(d1)
            except Exception:
                delta = 1.0 if S > K else 0.0
            total += pos * delta
        return total

    # --------------------------------------------------------
    # Fair value EWMA (持久化在 traderData 里)
    # warmup 期 (tick<100) α=0.10 快速对齐，之后 α=0.02 稳定追踪
    # --------------------------------------------------------
    WARMUP_TICKS = 100
    ALPHA_WARMUP = 0.10
    ALPHA_STEADY = 0.02

    def _ewma_update(self, current, new_obs, default, alpha):
        if current is None:
            current = default
        if new_obs is None:
            return current
        return (1 - alpha) * current + alpha * new_obs

    def _load_fair_state(self, trader_data):
        empty = {"hydrogel": None, "vev": None, "tick": 0}
        if not trader_data:
            return empty
        try:
            d = json.loads(trader_data)
            if not isinstance(d, dict):
                return empty
            return {
                "hydrogel": d.get("hydrogel"),
                "vev":      d.get("vev"),
                "tick":     int(d.get("tick", 0)),
            }
        except Exception:
            return empty

    def _save_fair_state(self, fair_state):
        try:
            return json.dumps(fair_state, separators=(",", ":"))
        except Exception:
            return ""

    # --------------------------------------------------------
    # 工具
    # --------------------------------------------------------
    def _get_mid(self, order_depth: OrderDepth):
        """Microprice: 用 top-of-book size 加权 — 比简单 mid 更准。"""
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

    # --------------------------------------------------------
    # 通用 mean-reversion + 5 层做市（layer 宽度按 std 缩放）
    # --------------------------------------------------------
    def _trade_mr(self, order_depth: OrderDepth, position: int, product: str,
                  fair: float, std: float, limit: int,
                  effective_position: int = None,
                  force_unwind: int = 0) -> List[Order]:
        """layers 在 ±0.15σ ±0.3σ ±0.5σ ±0.8σ ±1.2σ 处挂单
        take 阈值 0.5σ；rev take 1σ；panic 50% 仓位
        effective_position (optional): if provided, used for inventory state machine
        (long_heavy/short_heavy/panic) — actual `position` still used for buy_room/sell_room.
        force_unwind: 0 = normal; +1 = long stuck, aggressively cross BIDs to flatten;
                      -1 = short stuck, aggressively cross ASKs to flatten. When set,
                      we abandon all normal MM/take/layer logic for this tick and ONLY
                      emit unwind crosses, to free capacity ASAP."""
        orders: List[Order] = []
        if order_depth is None:
            return orders

        # === Force-unwind short-circuit ===
        # Cross the spread aggressively to drain stuck inventory. Skip all other
        # logic so we never simultaneously add new same-direction exposure.
        if force_unwind == +1 and position > 0:
            qty_to_unwind = min(FORCE_UNWIND_QTY_PER_TICK, position)
            if qty_to_unwind > 0 and order_depth.buy_orders:
                for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if qty_to_unwind <= 0:
                        break
                    vol = abs(order_depth.buy_orders[price])
                    qty = min(vol, qty_to_unwind)
                    if qty > 0:
                        orders.append(Order(product, int(price), -qty))
                        qty_to_unwind -= qty
            return orders
        elif force_unwind == -1 and position < 0:
            qty_to_unwind = min(FORCE_UNWIND_QTY_PER_TICK, -position)
            if qty_to_unwind > 0 and order_depth.sell_orders:
                for price in sorted(order_depth.sell_orders.keys()):
                    if qty_to_unwind <= 0:
                        break
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, qty_to_unwind)
                    if qty > 0:
                        orders.append(Order(product, int(price), qty))
                        qty_to_unwind -= qty
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

        # top-of-book imbalance: r=0.23-0.48 across products (out-of-sample verified)
        bv1 = abs(order_depth.buy_orders[best_bid]) if best_bid is not None else 0
        av1 = abs(order_depth.sell_orders[best_ask]) if best_ask is not None else 0
        I_top = (bv1 - av1) / (bv1 + av1) if (bv1 + av1) > 0 else 0.0

        buy_room = limit - position
        sell_room = limit + position

        # std 缩放尺度
        L1 = max(1, int(round(std * 0.15)))
        L2 = max(L1 + 1, int(round(std * 0.30)))
        L3 = max(L2 + 1, int(round(std * 0.50)))
        L4 = max(L3 + 1, int(round(std * 0.80)))
        L5 = max(L4 + 1, int(round(std * 0.80)))
        TAKE_T = max(2, int(round(std * 1.00)))   # 主动吃单阈值
        REV_T  = max(3, int(round(std * 2.25)))   # 反转跨价吃单阈值

        # 仓位状态 — delta-aware when effective_position is supplied
        eff = effective_position if effective_position is not None else position
        long_heavy   = eff >  int(limit * 0.55)
        short_heavy  = eff < -int(limit * 0.55)
        panic_long   = eff >  int(limit * 0.70)
        panic_short  = eff < -int(limit * 0.70)

        # === 主动吃单：仅在偏离 ≥ TAKE_T 时 ===
        # 公平区间外才吃单，不再无脑 fair±1
        take_buy_max  = fair - TAKE_T
        take_sell_min = fair + TAKE_T
        # 库存重时收缩 take 区间
        if long_heavy or panic_long:
            take_buy_max  = fair - max(TAKE_T, L4)   # 更便宜才买
            take_sell_min = fair - L1                 # 更早卖
        elif short_heavy or panic_short:
            take_buy_max  = fair + L1
            take_sell_min = fair + max(TAKE_T, L4)

        buy_done = sell_done = 0
        # I_top filter: block take only on strongly adverse imbalance
        # heavy_long/short already alters take edge; don't double-filter when reducing exposure
        allow_buy_take  = (I_top >= -0.5) or short_heavy or panic_short
        allow_sell_take = (I_top <=  0.5) or long_heavy  or panic_long
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

        # === 反转跨价吃单：偏离 ≥ 1σ 才考虑 ===
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

        # === 5 层挂单（按 std 缩放） ===
        f = int(round(fair))
        # 总挂量约 limit 的 50%（每边），留 50% 缓冲
        per_side = max(40, int(limit * 0.50))
        # Front-weighted: inner layers earn spread at top-of-book.
        # Live 462203 showed 0 passive fills (all aggressive crosses): outer-only L5
        # fell outside [best_bid+1, best_ask-1] clamp → no passive posting → -14.5K spread leak.
        layer_qtys = [int(per_side * w) for w in [0.3, 0.25, 0.2, 0.15, 0.1]]

        bid_layers = list(zip([f - L1, f - L2, f - L3, f - L4, f - L5], layer_qtys))
        ask_layers = list(zip([f + L1, f + L2, f + L3, f + L4, f + L5], layer_qtys))

        # 反转偏置：偏高 → 收紧 ask，外推 bid
        if deviation >= TAKE_T:
            ask_layers = list(zip([f, f + L1, f + L2, f + L3, f + L4], layer_qtys))
            bid_layers = list(zip([f - L2, f - L3, f - L4, f - L5], layer_qtys[:4]))
        elif deviation <= -TAKE_T:
            bid_layers = list(zip([f, f - L1, f - L2, f - L3, f - L4], layer_qtys))
            ask_layers = list(zip([f + L2, f + L3, f + L4, f + L5], layer_qtys[:4]))

        # 仓位保护
        if panic_long:
            ask_layers = [(f, per_side // 2), (f + L1, per_side // 2)]
            bid_layers = []
        elif long_heavy:
            # 保留全部 5 层 ask（包括外层接住 spike sell），仅缩量
            ask_layers = [(f + L1, per_side // 4), (f + L2, per_side // 4), (f + L3, per_side // 4),
                          (f + L4, per_side // 4), (f + L5, per_side // 4)]
            bid_layers = [(f - L4, per_side // 4)]   # 只在远端少量补仓
        elif panic_short:
            bid_layers = [(f, per_side // 2), (f - L1, per_side // 2)]
            ask_layers = []
        elif short_heavy:
            bid_layers = [(f - L1, per_side // 4), (f - L2, per_side // 4), (f - L3, per_side // 4),
                          (f - L4, per_side // 4), (f - L5, per_side // 4)]
            ask_layers = [(f + L4, per_side // 4)]

        rb = buy_room - buy_done
        rs = sell_room - sell_done
        # 防止 layer 跨价（关键 live bug fix：fair<<mid 时 inner ask 会 cross bid）
        max_bid = (best_ask - 1) if best_ask is not None else 10**9
        min_ask = (best_bid + 1) if best_bid is not None else 1
        for price, qty in bid_layers:
            if rb <= 0: break
            q = min(qty, rb)
            if q > 0 and 1 <= price <= max_bid:
                orders.append(Order(product, int(price), q))
                rb -= q
        for price, qty in ask_layers:
            if rs <= 0: break
            q = min(qty, rs)
            if q > 0 and price >= min_ask:
                orders.append(Order(product, int(price), -q))
                rs -= q

        return orders

    # --------------------------------------------------------
    # Voucher: IV smile 套利
    # --------------------------------------------------------
    def _trade_voucher(self, order_depth: OrderDepth, position: int, product: str,
                       K: int, S: float, force_unwind: int = 0,
                       cp_bias: float = 0.0) -> List[Order]:
        """cp_bias: additive shift to fair_price (price units), from contrarian
        own_trades signal. Positive when Mark 22/49 sold voucher to us → keep
        our long → fair UP. Negative when Mark 01/67 bought from us."""
        orders: List[Order] = []
        if order_depth is None or S is None:
            return orders
        if not order_depth.sell_orders and not order_depth.buy_orders:
            return orders

        limit = POSITION_LIMITS[product]

        # === Force-unwind short-circuit (vouchers) ===
        if force_unwind == +1 and position > 0:
            qty_to_unwind = min(FORCE_UNWIND_QTY_PER_TICK, position)
            if qty_to_unwind > 0 and order_depth.buy_orders:
                for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                    if qty_to_unwind <= 0:
                        break
                    vol = abs(order_depth.buy_orders[price])
                    qty = min(vol, qty_to_unwind)
                    if qty > 0:
                        orders.append(Order(product, int(price), -qty))
                        qty_to_unwind -= qty
            return orders
        elif force_unwind == -1 and position < 0:
            qty_to_unwind = min(FORCE_UNWIND_QTY_PER_TICK, -position)
            if qty_to_unwind > 0 and order_depth.sell_orders:
                for price in sorted(order_depth.sell_orders.keys()):
                    if qty_to_unwind <= 0:
                        break
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, qty_to_unwind)
                    if qty > 0:
                        orders.append(Order(product, int(price), qty))
                        qty_to_unwind -= qty
            return orders

        T = ROUND_TTE_DAYS / 252.0
        if T <= 0:
            return orders

        intrinsic = max(S - K, 0)

        # 深 ITM (VEV_4000): 用内在价值做市
        if K <= 4000:
            return self._floor_intrinsic_mm(order_depth, position, product, S, K, limit)
        # VEV_4500（中度 ITM，IV 反推不稳）和远 OTM (6000/6500，bid 经常 = 0)：
        # 不主动开仓，只在有持仓时被动平仓，避免占满仓位
        # 注：5000/5100/5200 已被 run() 路由到 _trade_mr (VEV_PROXY_PARAMS)，不会进到这里
        if K == 4500 or K >= 6000:
            return self._unwind_only(order_depth, position, product)

        m = math.log(K / S) / math.sqrt(T)
        fair_iv = smile_iv(m)
        fair_price = bs_call(S, K, T, fair_iv) + cp_bias  # CP bias shifts fair

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        mkt_mid = (best_ask + best_bid) / 2 if (best_ask and best_bid) else (best_ask or best_bid)
        if mkt_mid is None:
            return orders

        # 当前 IV 与 smile 拟合 IV 之差作为信号
        cur_iv = implied_vol(mkt_mid, S, K, T) if mkt_mid > intrinsic + 0.5 else None
        edge_threshold = SMILE_RESID_STD * EDGE_MULT.get(K, 1.0)

        buy_room = limit - position
        sell_room = limit + position
        buy_done = sell_done = 0

        # 仓位状态（和 _trade_mr 对齐）
        heavy_long  = position >  int(limit * 0.50)
        heavy_short = position < -int(limit * 0.50)
        panic_long  = position >  int(limit * 0.70)
        panic_short = position < -int(limit * 0.70)

        # 主动吃单：市场价远低于公允 → 买；远高于 → 卖
        # 仓位重时禁止同向加仓
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

        # IV 信号补充：明显偏离 smile 时反向挂单
        # heavy 时关闭同向、保留反向；panic 时全关
        if cur_iv is not None and not (panic_long or panic_short):
            iv_diff = cur_iv - fair_iv
            if iv_diff > edge_threshold and sell_room - sell_done > 0 and not heavy_short:
                px = max(int(math.ceil(fair_price + EAT_EDGE)), best_bid + 1 if best_bid else int(mkt_mid))
                qty = min(30, sell_room - sell_done)   # 单 tick 上限 30（原 50 易爆仓）
                if qty > 0:
                    orders.append(Order(product, px, -qty))
                    sell_done += qty
            elif iv_diff < -edge_threshold and buy_room - buy_done > 0 and not heavy_long:
                px = min(int(math.floor(fair_price - EAT_EDGE)), best_ask - 1 if best_ask else int(mkt_mid))
                qty = min(30, buy_room - buy_done)
                if qty > 0:
                    orders.append(Order(product, px, qty))
                    buy_done += qty

        # 围绕公允价的 2 层做市；panic 时只挂减仓侧
        f_int = int(round(fair_price))
        if f_int >= 1:
            mm_qty = max(5, min(30, limit // 10))
            mm_bid = f_int - 1
            mm_ask = f_int + 1
            if best_bid is not None: mm_bid = min(mm_bid, best_bid)
            if best_ask is not None: mm_ask = max(mm_ask, best_ask)
            allow_bid = not (heavy_long or panic_long)
            allow_ask = not (heavy_short or panic_short)
            if allow_bid and buy_room - buy_done > 0 and mm_bid >= 1:
                orders.append(Order(product, mm_bid, min(mm_qty, buy_room - buy_done)))
            if allow_ask and sell_room - sell_done > 0:
                orders.append(Order(product, mm_ask, -min(mm_qty, sell_room - sell_done)))

        # panic 强制减仓：在 mid 直接平 30 张
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

    def _unwind_only(self, order_depth, position, product):
        """不主动开仓；有持仓时按当前最优对手价平掉，没持仓就不挂单。"""
        orders: List[Order] = []
        if position == 0 or order_depth is None:
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

    def _floor_intrinsic_mm(self, order_depth, position, product, S, K, limit):
        """深 ITM (K<=4000)：时间价值≈0，按内在价值上下 0.5 做单。"""
        orders: List[Order] = []
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
