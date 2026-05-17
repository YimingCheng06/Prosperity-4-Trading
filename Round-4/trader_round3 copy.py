from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict, Tuple
import math
import json


# ============================================================
# Calibration constants — 用 colab_factor_mining.ipynb 训练后替换
# ============================================================
# IV smile: IV(m) = a*m^2 + b*m + c, where m = ln(K/S)/sqrt(T)
# Cleaned calibration: K=4500/4000/6000/6500 排除，Huber 鲁棒回归，per-day demean
SMILE_COEF = (0.030889, 0.004210, 0.192393)
SMILE_RESID_STD = 0.007357

VEV_FAIR = 5255.4        # recent-day mean (clean)
VEV_STD = 15.16          # intra-day std (no inter-day drift)
HYDROGEL_FAIR = 9989.4
HYDROGEL_STD = 31.92

# Round 3 起始 TTE = 5 天；如果题目算法环境里能拿到 round number，可动态算
# 简化：每 1_000_000 timestamp 当作一个 trading "day"，TTE 静态用 ROUND_TTE_DAYS
ROUND_TTE_DAYS = 5

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

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            mid = self._get_mid(order_depth)

            if product == "HYDROGEL_PACK":
                fair_eff, std_eff = self._blend_fair(sess, product, mid, HYDROGEL_FAIR, HYDROGEL_STD)
                orders = self._trade_mr(order_depth, position, product,
                                        fair=fair_eff, std=std_eff,
                                        limit=POSITION_LIMITS[product])
            elif product == "VELVETFRUIT_EXTRACT":
                fair_eff, std_eff = self._blend_fair(sess, product, mid, VEV_FAIR, VEV_STD)
                orders = self._trade_mr(order_depth, position, product,
                                        fair=fair_eff, std=std_eff,
                                        limit=POSITION_LIMITS[product])
            elif product == "VEV_4000":
                fair_eff, std_eff = self._blend_fair(sess, product, mid,
                                                    VEV_FAIR - 4000, VEV_STD)
                orders = self._trade_mr(order_depth, position, product,
                                        fair=fair_eff, std=std_eff,
                                        limit=POSITION_LIMITS[product])
            elif product.startswith("VEV_") and int(product.split("_")[1]) in VEV_PROXY_PARAMS:
                K = int(product.split("_")[1])
                fair_K, std_K = VEV_PROXY_PARAMS[K]
                fair_eff, std_eff = self._blend_fair(sess, product, mid, fair_K, std_K)
                orders = self._trade_mr(order_depth, position, product,
                                        fair=fair_eff, std=std_eff,
                                        limit=POSITION_LIMITS[product])
            elif product.startswith("VEV_"):
                K = int(product.split("_")[1])
                orders = self._trade_voucher(order_depth, position, product, K, vev_mid)
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
                  fair: float, std: float, limit: int) -> List[Order]:
        """layers 在 ±0.15σ ±0.3σ ±0.5σ ±0.8σ ±1.2σ 处挂单
        take 阈值 0.5σ；rev take 1σ；panic 50% 仓位"""
        orders: List[Order] = []
        if order_depth is None:
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

        # 仓位状态
        long_heavy   = position >  int(limit * 0.55)
        short_heavy  = position < -int(limit * 0.55)
        panic_long   = position >  int(limit * 0.70)
        panic_short  = position < -int(limit * 0.70)

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
                       K: int, S: float) -> List[Order]:
        orders: List[Order] = []
        if order_depth is None or S is None:
            return orders
        if not order_depth.sell_orders and not order_depth.buy_orders:
            return orders

        T = ROUND_TTE_DAYS / 252.0
        if T <= 0:
            return orders

        limit = POSITION_LIMITS[product]
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
        fair_price = bs_call(S, K, T, fair_iv)

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
