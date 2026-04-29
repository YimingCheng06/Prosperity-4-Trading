from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json


class Trader:
    """Round 4 v4 — base D + Ridge ML drift forecast as fair offset.

    Architecture:
      - Same warmup-freeze + drift correction (no hardcode for fair/std)
      - Same counterparty defensive take blocks (Variant D)
      - NEW: per-product Ridge linear models predict mid drift over k=200 ticks.
        Features: cum_Mark X, nf_Mark X_W{50,200}, vfe_mid, tp_roll, tp_dev,
        time_premium, rvol_{20,100}, ret_{20,50}, vfe_Itop. All maintained online.
        Forecast → fair offset (capped, scaled).

    Ridge models distilled from LightGBM SHAP top-8 features per product.
    Cross-day OOS sharpe: VEV_4000 +5.20, VEV_5300 +5.03, VEV_4500 +4.82, ...
    """

    BLOCK_BUY_TAKE_RULES = {
        "HYDROGEL_PACK": [
            ("Mark 14", "seller"),
            ("Mark 38", "buyer"),
        ],
    }
    BLOCK_SELL_TAKE_RULES = {
        "VELVETFRUIT_EXTRACT": [
            ("Mark 67", "buyer"),
            ("Mark 49", "seller"),
            ("Mark 22", "seller"),
            ("Mark 14", "seller"),
        ],
    }

    # ---- Ridge ML models (k=200 horizon, fitted on R4 d1+d2, OOS d3) ----
    # forecast = intercept + sum(w_i * (feat_i - mean_i) / std_i)
    # Use only k=200 models (k=50 is weaker for our directional bias use-case)
    RIDGE_MODELS = {
        "HYDROGEL_PACK": {
            "features": ['cum_Mark 38', 'cum_Mark 22', 'cum_Mark 14', 'vfe_mid', 'ret_50', 'nf_Mark 22_W200', 'rvol_100', 'nf_Mark 38_W200'],
            "means":    [29.3505, 1.2342, -30.5847, 5251.89, 0.1345, 0.0599, 2.15473, 0.75155],
            "stds":     [55.4517, 3.64439, 57.5522, 16.2255, 13.0837, 1.37004, 0.190386, 11.1829],
            "weights":  [0.204521, 0.0405558, -0.199625, -1.88671, -3.68233, -2.77053, 0.818551, 0.214875],
            "intercept": 0.58695,
            "oos_sharpe": 0.74,
        },
        "VELVETFRUIT_EXTRACT": {
            "features": ['cum_Mark 01', 'cum_Mark 14', 'cum_Mark 67', 'cum_Mark 49', 'nf_Mark 14_W200', 'cum_Mark 22', 'nf_Mark 14_W50', 'cum_Mark 55'],
            "means":    [-9.30345, 3.06435, 261.457, -167.573, 0.30315, -88.7671, 0.0762, 1.1223],
            "stds":     [30.6891, 45.6666, 149.002, 89.5672, 12.8302, 56.9058, 6.25939, 62.5973],
            "weights":  [-9.96836, -10.2268, 9.03769, 2.5252, -0.613313, 6.85988, -1.67806, -19.014],
            "intercept": 0.493925,
            "oos_sharpe": 1.44,
        },
        "VEV_4000": {
            "features": ['vfe_mid', 'cum_Mark 14', 'cum_Mark 38', 'rvol_100', 'tp_roll', 'nf_Mark 14_W200', 'nf_Mark 14_W50', 'nf_Mark 38_W200'],
            "means":    [5251.89, 6.28025, -5.9988, 1.40338, 0.0112163, 0.48575, 0.12085, -0.47575],
            "stds":     [16.2255, 15.3324, 15.0365, 0.270553, 0.0576161, 3.82165, 1.77726, 3.81742],
            "weights":  [-4.6818, -16.1313, -16.8745, -1.23443, -0.127049, 12.4819, -0.22928, 12.5134],
            "intercept": 0.496075,
            "oos_sharpe": 5.20,
        },
        "VEV_4500": {
            "features": ['vfe_mid', 'tp_roll', 'rvol_100', 'cum_Mark 22', 'ret_50', 'rvol_20', 'ret_20', 'tp_dev'],
            "means":    [5251.89, 0.0119738, 1.24671, -0.28145, 0.110875, 1.20662, 0.048925, -0.000823825],
            "stds":     [16.2255, 0.0562805, 0.189678, 0.449706, 6.6872, 0.40208, 4.25499, 0.763063],
            "weights":  [-4.7185, -0.234672, -1.396, 0.317086, 0.56837, 0.180284, 0.171889, -0.579777],
            "intercept": 0.495525,
            "oos_sharpe": 4.82,
        },
        "VEV_5000": {
            "features": ['tp_roll', 'vfe_mid', 'rvol_100', 'cum_Mark 22', 'tp_dev', 'ret_50', 'time_premium', 'ret_20'],
            "means":    [4.03176, 5251.89, 0.986795, -0.28145, -0.0229172, 0.098925, 4.01182, 0.044],
            "stds":     [1.33407, 16.2255, 0.0761362, 0.449706, 0.688972, 6.28352, 1.47463, 3.97219],
            "weights":  [5.61332, -5.8169, -0.272362, 0.233505, 3.15288, 0.620993, -8.15093, 0.211539],
            "intercept": 0.440475,
            "oos_sharpe": 2.34,
        },
        "VEV_5100": {
            "features": ['vfe_mid', 'tp_roll', 'tp_dev', 'rvol_100', 'time_premium', 'ret_20', 'cum_Mark 22', 'ret_50'],
            "means":    [5251.89, 14.3163, -0.0602744, 0.864531, 14.2623, 0.0353, -0.28145, 0.078475],
            "stds":     [16.2255, 3.44787, 1.33563, 0.0693068, 3.61888, 3.51842, 0.449706, 5.56273],
            "weights":  [-4.72233, 6.42044, 1.99439, -0.421263, -7.70529, 0.137462, 0.288297, 0.134013],
            "intercept": 0.352575,
            "oos_sharpe": 2.35,
        },
        "VEV_5200": {
            "features": ['cum_Mark 14', 'tp_roll', 'vfe_mid', 'time_premium', 'cum_Mark 22', 'rvol_100', 'tp_dev', 'nf_Mark 14_W200'],
            "means":    [9.89955, 42.8005, 5251.89, 42.6953, -10.181, 0.690128, -0.110757, 0.47],
            "stds":     [9.62242, 6.94258, 16.2255, 7.31875, 9.69751, 0.0730365, 2.79479, 1.51043],
            "weights":  [4.38645, 6.08426, -3.88868, -7.36022, 3.84712, -0.0497584, 1.90312, -0.405941],
            "intercept": 0.224725,
            "oos_sharpe": 1.99,
        },
        "VEV_5300": {
            "features": ['tp_roll', 'time_premium', 'cum_Mark 01', 'cum_Mark 22', 'tp_dev', 'vfe_mid', 'cum_Mark 14', 'rvol_100'],
            "means":    [45.656, 45.6936, 58.3225, -74.3147, 0.0217512, 5251.89, 15.7108, 0.497318],
            "stds":     [6.06506, 6.43095, 32.5376, 44.036, 2.8725, 16.2255, 13.8416, 0.0536195],
            "weights":  [0.37275, -0.258208, -0.925371, -0.85298, 0.503303, -2.2666, 0.646954, 0.459361],
            "intercept": 0.1048,
            "oos_sharpe": 5.03,
        },
    }

    # ML usage tunables
    RIDGE_FILTER_THRESH = 0.0      # take filter disabled (hurt PnL across all settings)
    RIDGE_TARGET_SCALE = 0.15      # forecast/std → target_pos fraction
    RIDGE_TARGET_MAX_FRAC = 0.30   # cap target_pos at 30% of limit
    RIDGE_WARMUP_TICKS = 100       # smaller for live engagement (was 200)
    RIDGE_MIN_SHARPE = 0.0         # use all 8 models
    ROLL_W_SHORT = 50
    ROLL_W_LONG = 200
    ACTORS = ("Mark 01", "Mark 14", "Mark 22", "Mark 38", "Mark 49", "Mark 55", "Mark 67")

    # ---- Warmup / freeze ----
    # CRITICAL: live submission = 1000 ticks (1/10 day). Must freeze well before.
    # Backtest 10K ticks. WARMUP=300 works in both; trade-off is less accurate
    # frozen mean → larger STD_INFLATE + aggressive drift correction compensate.
    WARMUP_FREEZE_N = 300
    DRIFT_CORRECT_START = 400
    TRIM_Q = 0.05
    STD_INFLATE = 1.70

    # ---- Drift correction ----
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
    def _detect_block(self, product: str, market_trades) -> tuple:
        """Returns (block_buy, block_sell) booleans based on this-tick triggers."""
        block_buy = False
        block_sell = False
        if not market_trades:
            return block_buy, block_sell
        buy_rules = self.BLOCK_BUY_TAKE_RULES.get(product, [])
        sell_rules = self.BLOCK_SELL_TAKE_RULES.get(product, [])
        for trade in market_trades:
            buyer = getattr(trade, "buyer", None)
            seller = getattr(trade, "seller", None)
            for actor, role in buy_rules:
                if (role == "buyer" and buyer == actor) or (role == "seller" and seller == actor):
                    block_buy = True
            for actor, role in sell_rules:
                if (role == "buyer" and buyer == actor) or (role == "seller" and seller == actor):
                    block_sell = True
        return block_buy, block_sell

    # ============================================================
    # Online feature state for Ridge models
    # ============================================================
    def _ridge_state_init(self, sess: dict, product: str) -> dict:
        """Per-product feature state. Lazy init on first call."""
        key = f"_ml_{product}"
        if key not in sess:
            sess[key] = {
                # rolling buffers (lists used as deques; capped at ROLL_W_LONG)
                "mids": [],          # for ret_*, rvol_*
                "midprice_raw": [],  # for ret_50 etc — same as mids
                "tp": [],            # voucher: time premium series for tp_roll
                # per-actor cumulative net flow (signed)
                "cum": {a: 0.0 for a in self.ACTORS},
                # per-actor net-flow rolling buffers (size up to ROLL_W_LONG)
                "nf_buf": {a: [] for a in self.ACTORS},
                # last frozen tp_roll for use as-feature (rolling 200 mean shifted by 1)
                "ticks": 0,
            }
        return sess[key]

    def _ridge_update_mid_tp(self, sess: dict, product: str, mid, vfe_microprice):
        """Phase 1: append current mid/tp (training: ret_K uses mid_t)."""
        ps = self._ridge_state_init(sess, product)
        if mid is not None:
            ps["mids"].append(float(mid))
            if len(ps["mids"]) > self.ROLL_W_LONG + 60:
                ps["mids"] = ps["mids"][-(self.ROLL_W_LONG + 60):]
        if product.startswith("VEV_") and mid is not None and vfe_microprice is not None:
            try:
                K = int(product.split("_")[1])
                intrinsic = max(vfe_microprice - K, 0.0)
                tp_now = float(mid) - intrinsic
                ps["tp"].append(tp_now)
                if len(ps["tp"]) > self.ROLL_W_LONG + 60:
                    ps["tp"] = ps["tp"][-(self.ROLL_W_LONG + 60):]
            except Exception:
                pass

    def _ridge_update_flow(self, sess: dict, product: str, market_trades):
        """Phase 2: update cum/nf AFTER feature compute (training: nf/cum used shift(1))."""
        ps = self._ridge_state_init(sess, product)
        per_tick_nf = {a: 0 for a in self.ACTORS}
        for trade in (market_trades or []):
            buyer = getattr(trade, "buyer", None)
            seller = getattr(trade, "seller", None)
            qty = int(getattr(trade, "quantity", 0))
            if buyer in per_tick_nf:
                per_tick_nf[buyer] += qty
            if seller in per_tick_nf:
                per_tick_nf[seller] -= qty
        for a in self.ACTORS:
            ps["cum"][a] += per_tick_nf[a]
            buf = ps["nf_buf"][a]
            buf.append(per_tick_nf[a])
            if len(buf) > self.ROLL_W_LONG:
                ps["nf_buf"][a] = buf[-self.ROLL_W_LONG:]
        ps["ticks"] += 1

    def _ridge_compute_features(self, sess: dict, product: str,
                                 vfe_microprice, vfe_Itop) -> dict:
        """Compute current feature dict from accumulated state. Pure read; no update."""
        ps = sess.get(f"_ml_{product}")
        if ps is None:
            return {}
        f = {}
        # mid-derived features (use values BEFORE current tick if possible — but here
        # we use the most recent values, accepting 1-tick alignment difference)
        mids = ps["mids"]
        if len(mids) >= 51:
            f["ret_1"]  = mids[-1] - mids[-2]
            f["ret_3"]  = mids[-1] - mids[-4]  if len(mids)>=4 else 0.0
            f["ret_5"]  = mids[-1] - mids[-6]  if len(mids)>=6 else 0.0
            f["ret_10"] = mids[-1] - mids[-11] if len(mids)>=11 else 0.0
            f["ret_20"] = mids[-1] - mids[-21] if len(mids)>=21 else 0.0
            f["ret_50"] = mids[-1] - mids[-51] if len(mids)>=51 else 0.0
            # rvol: std of mid diffs
            d20 = [mids[i] - mids[i-1] for i in range(max(1, len(mids)-20), len(mids))]
            if len(d20) >= 2:
                m_d = sum(d20)/len(d20)
                v_d = sum((x-m_d)**2 for x in d20)/len(d20)
                f["rvol_20"] = v_d ** 0.5
            d100 = [mids[i] - mids[i-1] for i in range(max(1, len(mids)-100), len(mids))]
            if len(d100) >= 2:
                m_d = sum(d100)/len(d100)
                v_d = sum((x-m_d)**2 for x in d100)/len(d100)
                f["rvol_100"] = v_d ** 0.5

        # Voucher-only: tp_roll, tp_dev, time_premium
        if product.startswith("VEV_"):
            tp = ps["tp"]
            if len(tp) >= 21:
                # rolling 200 mean (or whatever's available, min_periods=20), SHIFTED BY 1
                window = tp[-200:-1] if len(tp) >= 21 else tp[:-1]
                if len(window) >= 20:
                    f["tp_roll"] = sum(window) / len(window)
            if "tp_roll" in f and tp:
                f["time_premium"] = tp[-1]
                f["tp_dev"] = tp[-1] - f["tp_roll"]

        # Underlying VFE features for vouchers
        if vfe_microprice is not None:
            f["vfe_mid"] = float(vfe_microprice)
        if vfe_Itop is not None:
            f["vfe_Itop"] = float(vfe_Itop)

        # Counterparty cumulative & windowed
        for a in self.ACTORS:
            f[f"cum_{a}"] = float(ps["cum"][a])
            buf = ps["nf_buf"][a]
            if len(buf) > 0:
                f[f"nf_{a}_W{self.ROLL_W_SHORT}"] = float(sum(buf[-self.ROLL_W_SHORT:]))
                f[f"nf_{a}_W{self.ROLL_W_LONG}"]  = float(sum(buf[-self.ROLL_W_LONG:]))

        return f

    def _ridge_forecast(self, product: str, features: dict):
        """Compute drift forecast for product. Returns None if any required feature missing."""
        model = self.RIDGE_MODELS.get(product)
        if model is None:
            return None
        names = model["features"]
        means = model["means"]
        stds = model["stds"]
        weights = model["weights"]
        intercept = model["intercept"]
        s = float(intercept)
        for i, n in enumerate(names):
            v = features.get(n)
            if v is None:
                return None     # missing feature → don't bias
            z = (float(v) - means[i]) / stds[i]
            s += weights[i] * z
        return s

    def run(self, state: TradingState):
        result: Dict[str, List[Order]] = {}
        sess = self._load_session(getattr(state, "traderData", "") or "")
        self._check_day_reset(sess, state.timestamp)
        market_trades = getattr(state, "market_trades", {}) or {}

        # --- pre-pass: compute VFE microprice + I_top for cross-product features ---
        vfe_depth = state.order_depths.get("VELVETFRUIT_EXTRACT")
        vfe_microprice = self._get_mid(vfe_depth) if vfe_depth else None
        vfe_Itop = None
        if vfe_depth and vfe_depth.buy_orders and vfe_depth.sell_orders:
            bb = max(vfe_depth.buy_orders.keys())
            ba = min(vfe_depth.sell_orders.keys())
            bv = abs(vfe_depth.buy_orders[bb]); av = abs(vfe_depth.sell_orders[ba])
            if bv + av > 0:
                vfe_Itop = (bv - av) / (bv + av)

        for product, order_depth in state.order_depths.items():
            position = state.position.get(product, 0)
            mid = self._get_mid(order_depth)

            if product in self.SKIP_PRODUCTS:
                result[product] = self._unwind_only(order_depth, position, product)
                continue

            if product not in self.POSITION_LIMITS:
                result[product] = []
                continue

            # Phase 1: update mids/tp (so ret_K, time_premium include current mid)
            self._ridge_update_mid_tp(sess, product, mid, vfe_microprice)

            fair, std, ready = self._update_and_get_fair(sess, product, mid)

            # Compute features NOW: mids include current (correct for ret/rvol),
            # nf/cum still hold past-only (correct for shift(1) alignment).
            ridge_block_buy = False
            ridge_block_sell = False
            ridge_target_pos = 0
            ml_state = sess.get(f"_ml_{product}")
            model = self.RIDGE_MODELS.get(product)
            if (ml_state is not None
                and model is not None
                and model.get("oos_sharpe", 0) >= self.RIDGE_MIN_SHARPE
                and ml_state.get("ticks", 0) >= self.RIDGE_WARMUP_TICKS
                and ready):
                features = self._ridge_compute_features(sess, product,
                                                         vfe_microprice, vfe_Itop)
                forecast = self._ridge_forecast(product, features)
                if forecast is not None:
                    if self.RIDGE_FILTER_THRESH > 0:
                        thresh = self.RIDGE_FILTER_THRESH * std
                        if forecast <= -thresh:
                            ridge_block_buy = True
                        elif forecast >= thresh:
                            ridge_block_sell = True
                    if self.RIDGE_TARGET_SCALE > 0:
                        # forecast in price units → divide by std to get z-score
                        # then map to position fraction
                        z = forecast / max(std, 1.0)
                        limit = self.POSITION_LIMITS[product]
                        max_t = int(limit * self.RIDGE_TARGET_MAX_FRAC)
                        ridge_target_pos = int(max(-max_t, min(max_t,
                                                z * self.RIDGE_TARGET_SCALE * max_t)))

            # Phase 2: update flow AFTER features computed (matches shift(1))
            self._ridge_update_flow(sess, product, market_trades.get(product, []))

            if not ready:
                result[product] = []
                continue

            cp_block_buy, cp_block_sell = self._detect_block(product, market_trades.get(product, []))
            block_buy = cp_block_buy or ridge_block_buy
            block_sell = cp_block_sell or ridge_block_sell

            orders = self._trade_mr(
                order_depth, position, product,
                fair=fair, std=std,
                limit=self.POSITION_LIMITS[product],
                block_buy_take=block_buy, block_sell_take=block_sell,
                target_position=ridge_target_pos,
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
                  fair: float, std: float, limit: int,
                  block_buy_take: bool = False, block_sell_take: bool = False,
                  target_position: int = 0) -> List[Order]:
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

        # Heavy/panic measured RELATIVE to ridge target_position (Ridge bias)
        rel_pos = position - target_position
        long_heavy   = rel_pos >  int(limit * self.HEAVY_FRAC)
        short_heavy  = rel_pos < -int(limit * self.HEAVY_FRAC)
        panic_long   = rel_pos >  int(limit * self.PANIC_FRAC)
        panic_short  = rel_pos < -int(limit * self.PANIC_FRAC)

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
        # Counterparty defensive blocks override I_top, but never block forced reduction
        # (heavy/panic still take to reduce exposure)
        if block_buy_take and not (short_heavy or panic_short):
            allow_buy_take = False
        if block_sell_take and not (long_heavy or panic_long):
            allow_sell_take = False
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
