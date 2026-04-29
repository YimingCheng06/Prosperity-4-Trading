from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json


class Trader:
    """Round 4 base trader — no hardcode, online warmup-freeze + drift correction.

    Architecture:
      Phase 1 (t < WARMUP_FREEZE_N): silent. Accumulate per-product mid → trimmed mean/std.
      Phase 2 (t >= WARMUP_FREEZE_N): freeze fair/std. Run R3-style take-and-revert.
      Phase 2 + drift (t >= DRIFT_CORRECT_START): if running session-mean drifts >GATE σ from frozen,
          blend frozen toward running mean (R3 SESSION_MEAN_BLEND mechanic).

    No hardcoded per-product fair/std/coef — all derived from current session's observed mids.
    """

    # ---- Warmup / freeze ----
    # CRITICAL: live = 1000 ticks. WARMUP must be << 1000 (here 300).
    WARMUP_FREEZE_N = 300
    DRIFT_CORRECT_START = 400
    TRIM_Q = 0.05
    STD_INFLATE = 1.70

    # ---- Drift correction (aggressive to compensate for noisy short-warmup mean) ----
    DRIFT_RAMP_N = 300
    DRIFT_MAX_W = 0.8
    DRIFT_MAX_W_STD = 0.7
    DRIFT_GATE = 0.0

    # ---- Trading hyperparams (algo constants, not product-specific) ----
    TAKE_T_MULT = 1.00               # take threshold in σ
    REV_T_MULT  = 2.25               # cross-fair revert threshold in σ
    HEAVY_FRAC  = 0.55               # |pos| > HEAVY*limit → heavy regime
    PANIC_FRAC  = 0.70
    PER_SIDE_FRAC = 0.50
    LAYER_QTY_WEIGHTS = (0.30, 0.25, 0.20, 0.15, 0.10)
    LAYER_WIDTHS_SIGMA = (0.15, 0.30, 0.50, 0.80, 0.80)
    I_TOP_BLOCK = 0.5                # |I_top| beyond this blocks take on adverse side

    # ---- Position limits (rule-given) ----
    POSITION_LIMITS = {
        "HYDROGEL_PACK": 200,
        "VELVETFRUIT_EXTRACT": 200,
    }
    for _K in (4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500):
        POSITION_LIMITS[f"VEV_{_K}"] = 300
    del _K

    # Dead products (R4 data: std=0 all 3 days, mid stuck at 0.5) → unwind only
    SKIP_PRODUCTS = {"VEV_6000", "VEV_6500"}

    # ============================================================
    # session state (persisted via traderData)
    # ============================================================
    def _load_session(self, trader_data: str) -> dict:
        if not trader_data:
            return {}
        try:
            d = json.loads(trader_data)
            return d if isinstance(d, dict) else {}
        except Exception:
            return {}

    def _save_session(self, sess: dict) -> str:
        try:
            return json.dumps(sess, separators=(",", ":"))
        except Exception:
            return ""

    def _check_day_reset(self, sess: dict, current_ts: int):
        last_ts = sess.get("_last_ts", -1)
        if current_ts == 0 or current_ts < last_ts:
            keys = [k for k in sess.keys() if k != "_last_ts"]
            for k in keys:
                del sess[k]
        sess["_last_ts"] = current_ts

    # ============================================================
    # Online warmup-freeze + drift correction
    # ============================================================
    def _update_and_get_fair(self, sess: dict, product: str, mid):
        """Accumulate mid, freeze at N, then drift-correct. Returns (fair, std, ready).

        ready=False → still in warmup, caller should not trade actively.
        """
        ps = sess.setdefault(product, {
            "warm_mids": [],   # raw mid samples during warmup (for trimmed stats)
            "frozen_m": None,
            "frozen_s": None,
            "post_sum": 0.0,
            "post_ssq": 0.0,
            "post_n": 0,
        })

        if ps["frozen_m"] is None:
            # Phase 1: accumulate
            if mid is not None:
                ps["warm_mids"].append(float(mid))
            if len(ps["warm_mids"]) >= self.WARMUP_FREEZE_N:
                arr = sorted(ps["warm_mids"])
                n = len(arr)
                lo = int(n * self.TRIM_Q)
                hi = n - lo
                trimmed = arr[lo:hi] if hi > lo else arr
                m = sum(trimmed) / len(trimmed)
                v = sum((x - m) * (x - m) for x in trimmed) / len(trimmed)
                s = (v ** 0.5) * self.STD_INFLATE
                ps["frozen_m"] = m
                ps["frozen_s"] = max(s, 1.0)
                ps["warm_mids"] = []           # free memory
            return None, None, False

        # Phase 2: frozen, possibly drift-corrected
        frozen_m = ps["frozen_m"]
        frozen_s = ps["frozen_s"]

        # Accumulate post-warmup mean & sum-of-squares (for drift correction)
        if mid is not None:
            mf = float(mid)
            ps["post_sum"] += mf
            ps["post_ssq"] += mf * mf
            ps["post_n"] += 1

        post_n = ps["post_n"]
        ramp_start = max(0, self.DRIFT_CORRECT_START - self.WARMUP_FREEZE_N)
        if post_n < ramp_start + 1:
            return frozen_m, frozen_s, True

        post_mean = ps["post_sum"] / post_n
        post_var = max(0.0, ps["post_ssq"] / post_n - post_mean * post_mean)
        post_std = max(post_var ** 0.5, 1.0)

        offset = abs(post_mean - frozen_m)
        gate = self.DRIFT_GATE * frozen_s

        ramp = min(1.0, (post_n - ramp_start) / max(1, self.DRIFT_RAMP_N))
        w_base = self.DRIFT_MAX_W * ramp

        if gate <= 0:
            w_mean = w_base
        elif offset < gate:
            w_mean = 0.0
        else:
            w_mean = min(w_base, w_base * (offset - gate) / gate)

        fair_eff = (1 - w_mean) * frozen_m + w_mean * post_mean

        # std blend: gateless ramp toward post_std (better intraday vol estimate)
        w_std = ramp * self.DRIFT_MAX_W_STD
        std_eff = max(0.5 * frozen_s, (1 - w_std) * frozen_s + w_std * post_std)

        return fair_eff, std_eff, True

    # ============================================================
    # main entry
    # ============================================================
    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        sess = self._load_session(getattr(state, "traderData", "") or "")
        self._check_day_reset(sess, state.timestamp)

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            mid = self._get_mid(order_depth)

            if product in self.SKIP_PRODUCTS:
                # Dead OTM voucher: unwind any leftover, no new exposure
                result[product] = self._unwind_only(order_depth, position, product)
                continue

            if product not in self.POSITION_LIMITS:
                # Unknown symbol — skip safely
                result[product] = []
                continue

            fair, std, ready = self._update_and_get_fair(sess, product, mid)

            if not ready:
                # Phase 1 (warmup): silent, no orders
                result[product] = []
                continue

            orders = self._trade_mr(
                order_depth, position, product,
                fair=fair, std=std,
                limit=self.POSITION_LIMITS[product],
            )
            result[product] = orders

        return result, 0, self._save_session(sess)

    # ============================================================
    # helpers
    # ============================================================
    def _get_mid(self, order_depth: OrderDepth):
        """Microprice: top-of-book size-weighted."""
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

    def _unwind_only(self, order_depth, position, product):
        """Liquidate at best price; no new exposure."""
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

    # ============================================================
    # mean-reversion + 5-layer MM (R3 logic, std-scaled)
    # ============================================================
    def _trade_mr(self, order_depth: OrderDepth, position: int, product: str,
                  fair: float, std: float, limit: int) -> List[Order]:
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
            mid = best_ask - max(1, int(std * 0.10))

        deviation = mid - fair

        bv1 = abs(order_depth.buy_orders[best_bid]) if best_bid is not None else 0
        av1 = abs(order_depth.sell_orders[best_ask]) if best_ask is not None else 0
        I_top = (bv1 - av1) / (bv1 + av1) if (bv1 + av1) > 0 else 0.0

        buy_room = limit - position
        sell_room = limit + position

        # Layer widths in σ
        w1, w2, w3, w4, w5 = self.LAYER_WIDTHS_SIGMA
        L1 = max(1, int(round(std * w1)))
        L2 = max(L1 + 1, int(round(std * w2)))
        L3 = max(L2 + 1, int(round(std * w3)))
        L4 = max(L3 + 1, int(round(std * w4)))
        L5 = max(L4 + 1, int(round(std * w5)))
        TAKE_T = max(2, int(round(std * self.TAKE_T_MULT)))
        REV_T  = max(3, int(round(std * self.REV_T_MULT)))

        long_heavy   = position >  int(limit * self.HEAVY_FRAC)
        short_heavy  = position < -int(limit * self.HEAVY_FRAC)
        panic_long   = position >  int(limit * self.PANIC_FRAC)
        panic_short  = position < -int(limit * self.PANIC_FRAC)

        # === active take ===
        take_buy_max  = fair - TAKE_T
        take_sell_min = fair + TAKE_T
        if long_heavy or panic_long:
            take_buy_max  = fair - max(TAKE_T, L4)
            take_sell_min = fair - L1
        elif short_heavy or panic_short:
            take_buy_max  = fair + L1
            take_sell_min = fair + max(TAKE_T, L4)

        buy_done = sell_done = 0
        allow_buy_take  = (I_top >= -self.I_TOP_BLOCK) or short_heavy or panic_short
        allow_sell_take = (I_top <=  self.I_TOP_BLOCK) or long_heavy  or panic_long
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

        # === reversion cross-take (deviation >= REV_T) ===
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

        # === 5-layer passive MM ===
        f = int(round(fair))
        per_side = max(40, int(limit * self.PER_SIDE_FRAC))
        layer_qtys = [int(per_side * w) for w in self.LAYER_QTY_WEIGHTS]

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
