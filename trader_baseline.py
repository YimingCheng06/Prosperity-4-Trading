"""
IMC Prosperity Round 5 — Baseline Trader
========================================

Single-file submission for the Prosperity platform.

Strategy summary (per-category):
  Tier 1 (GOLD):
    - PEBBLES        : linear conservation XS+S+M+L+XL == 50000 -> basket arb
    - SNACK_PACKS    : two pair-baskets (CHOC+VAN, STRAW+RASP) z-score reversion
                       PISTACHIO -> passive MM fallback
  Tier 2 (SILVER):
    - SLEEP_PODS     : POLY ~ 0.964*COTTON, LAMB ~ 0.401*NYLON pair reversion
                       SUEDE -> passive MM fallback
    - ROBOTS         : DISHES vs equal-weight basket of other 4
    - UV_VISORS      : AMBER + 0.53 * MAGENTA pair reversion
  Tier 3 (BRONZE) — small size, |pos|<=3:
    - GALAXY_SOUNDS  : SOLAR_FLAMES + 0.28 * SOLAR_WINDS
    - MICROCHIPS     : SQUARE + 2.15 * RECTANGLE
    - TRANSLATORS    : ECLIPSE_CHARCOAL - 0.287 * VOID_BLUE
    - PANELS         : PANEL_1X2 + 0.47 * PANEL_2X2
  Tier 4 (NOPE):
    - OXYGEN_SHAKES  : skipped entirely (sign-flipping hedge ratios)

API:
  Trader().run(state) -> (orders_dict, conversions_int, traderData_str)
"""

from datamodel import OrderDepth, TradingState, Order, Symbol, Trade, Listing  # noqa: F401
from typing import Dict, List, Tuple, Optional, Iterable
import json
import math
import statistics


# =============================================================================
#                                 CONSTANTS
# =============================================================================
# All tunables live here so the user can sweep them in Colab/Nbis.

LOG_LEVEL = "OFF"  # "OFF" | "INFO" | "DEBUG"

# --- Global ------------------------------------------------------------------
POSITION_LIMIT = 10                # hard exchange-side limit per product
DEFAULT_MAX_HISTORY = 2000         # how many points to retain in rolling deques

# --- PEBBLES (downgraded — basket take-arb NOT exploitable; passive only) ----
# DEEP-DIVE FINDING: market makers also enforce sum=50000 at top of book
# (sum_ask ~ 50032, sum_bid ~ 49968, round-trip cost ~65 ticks vs max mid wedge
# 35 ticks). Take-the-book wedge is positive only 1.49% of the time with max
# 4 ticks of edge. Joint passive fill across all 5 legs = 0%.
# Strategy: per-leg passive quotes CONDITIONED on residual sign — no basket take.
PEBBLES = [
    "PEBBLES_XS", "PEBBLES_S", "PEBBLES_M", "PEBBLES_L", "PEBBLES_XL",
]
PEBBLES_TARGET_SUM = 50000          # mechanical conservation constant
# 2026-04-30 (rolled back): single-leg cross-spread alpha was net negative on
# the platform — rolling-mean lag plus full-spread cost on each fire created
# the late-day chop visible in the user's PnL trace. Reverted to the original
# residual-conditioned passive MM on all 5 legs. The single-leg idea is still
# saved in disruptive_ideas memory for a future passive variant.
THRESHOLD_PEBBLES_TAKE = 9999
PEBBLES_RESID_TRIGGER = 1           # |sum_mid - 50000| > 1 -> bias passive quotes
PEBBLES_PASSIVE_SIZE = 2            # size for residual-conditioned passive quotes
PEBBLES_UNWIND_SIZE = 2             # size for inventory unwind when |position| > 0

# --- SNACK_PACKS (Tier 1 — Gold) ---------------------------------------------
CLASSICS_LEGS = ["SNACKPACK_CHOCOLATE", "SNACKPACK_VANILLA"]
BERRIES_LEGS = ["SNACKPACK_STRAWBERRY", "SNACKPACK_RASPBERRY"]
SNACK_FALLBACK_MM = ["SNACKPACK_PISTACHIO"]

# 2026-04-30 grid: SNACK_Z_ENTER 1.5->2.0 (less aggressive), windows 500->800
# (more stable mu/sd) — Snack Packs was -220k, Z=1.5 was overfiring.
CLASSICS_WINDOW = 800
BERRIES_WINDOW = 800
SNACK_Z_ENTER = 2.0
SNACK_Z_EXIT = 0.3
SNACK_Z_STOP = 3.0
SNACK_MAX_LEG_SIZE = POSITION_LIMIT  # +-10

# 2026-04-30 ROLLED BACK: inter-basket cross-spread (Tier A #2) cost -4k on
# the platform (6.5k -> 2.5k). Cross-basket signal isn't actually MR on live
# data despite within-basket anti-correlation. Passive quotes still got hit
# adversely when both baskets trended together.

# --- SLEEP_PODS (Tier 2 — Silver) --------------------------------------------
SLEEP_POLY = "SLEEP_POD_POLYESTER"
SLEEP_COTTON = "SLEEP_POD_COTTON"
SLEEP_LAMB = "SLEEP_POD_LAMB_WOOL"
SLEEP_NYLON = "SLEEP_POD_NYLON"
SLEEP_SUEDE = "SLEEP_POD_SUEDE"

# 2026-04-30 OLS refit on Day 1+2+3 (n=20000):
#   POLY_COTTON: 0.964 -> 1.213 (R^2=0.630)  LARGE — was driving Sleep Pods -77k
#   LAMB_NYLON:  0.401 -> 0.592 (R^2=0.206)
HEDGE_POLY_COTTON = 1.213
HEDGE_LAMB_NYLON = 0.592

SLEEP_WINDOW = 1500                 # 2026-04-30 grid: 1000 -> 1500
SLEEP_Z_ENTER = 2.0
SLEEP_Z_EXIT = 0.3
SLEEP_Z_STOP = 3.5
SLEEP_MAX_LEG_SIZE = POSITION_LIMIT

# --- ROBOTS (Tier 2 — Silver) ------------------------------------------------
ROBOT_DISHES = "ROBOT_DISHES"
ROBOTS_OTHER = ["ROBOT_VACUUMING", "ROBOT_MOPPING", "ROBOT_LAUNDRY", "ROBOT_IRONING"]
# 2026-04-30 grid: ROBOTS_WINDOW 1000->500 (faster), Z_ENTER 2.0->2.5 (less firing).
# Robots was -706k — biggest single hole. These are safer, but if PnL still bleeds
# user has flagged we may need to disable Robots entirely.
ROBOTS_WINDOW = 500
ROBOTS_Z_ENTER = 2.5
ROBOTS_Z_EXIT = 0.3
ROBOTS_Z_STOP = 3.5
ROBOTS_MAX_LEG_SIZE = POSITION_LIMIT

# --- UV_VISORS (Tier 2 — Silver) ---------------------------------------------
VISOR_AMBER = "UV_VISOR_AMBER"
VISOR_MAGENTA = "UV_VISOR_MAGENTA"
VISORS_OTHER = ["UV_VISOR_YELLOW", "UV_VISOR_ORANGE", "UV_VISOR_RED"]
HEDGE_AMBER_MAGENTA = 1.238         # 2026-04-30 OLS refit (R^2=0.798) — was 0.53,
                                    # this 2.3x change explains UV Visors -286k
VISORS_WINDOW = 1000
VISORS_Z_ENTER = 2.0
VISORS_Z_EXIT = 0.3
VISORS_Z_STOP = 3.5
VISORS_MAX_LEG_SIZE = POSITION_LIMIT

# --- BRONZE pairs ------------------------------------------------------------
BRONZE_POSITION_CAP = 3            # |pos| <= 3 for all bronze products
BRONZE_Z_ENTER = 2.0               # 2026-04-30 grid: 2.5 -> 2.0 (valid_sharpe=0.27)
BRONZE_Z_EXIT = 0.3
BRONZE_Z_STOP = 4.0

# Galaxy
GALAXY_FLAMES = "GALAXY_SOUNDS_SOLAR_FLAMES"
GALAXY_WINDS = "GALAXY_SOUNDS_SOLAR_WINDS"
GALAXY_BLACK = "GALAXY_SOUNDS_BLACK_HOLES"        # avoided (trending)
GALAXY_OTHER = ["GALAXY_SOUNDS_DARK_MATTER", "GALAXY_SOUNDS_PLANETARY_RINGS"]  # MM-fallback only
HEDGE_FLAMES_WINDS = 0.340          # 2026-04-30 OLS refit (R^2=0.103, weak)
GALAXY_WINDOW = 1500

# Microchips
CHIP_SQUARE = "MICROCHIP_SQUARE"
CHIP_RECT = "MICROCHIP_RECTANGLE"
CHIPS_OTHER = ["MICROCHIP_CIRCLE", "MICROCHIP_OVAL", "MICROCHIP_TRIANGLE"]   # MM-fallback only
HEDGE_SQUARE_RECT = 2.145           # 2026-04-30 OLS refit (R^2=0.859, strong)
CHIPS_WINDOW = 1500

# Translators
TRANS_CHARCOAL = "TRANSLATOR_ECLIPSE_CHARCOAL"
TRANS_VOIDBLUE = "TRANSLATOR_VOID_BLUE"
TRANS_OTHER: List[str] = [
    "TRANSLATOR_SPACE_GRAY", "TRANSLATOR_ASTRO_BLACK", "TRANSLATOR_GRAPHITE_MIST",
]
HEDGE_CHARCOAL_VOIDBLUE = 0.457     # 2026-04-30 OLS refit (R^2=0.432) — was 0.287
TRANS_WINDOW = 1500

# Panels
PANEL_LEG_A = "PANEL_1X2"
PANEL_LEG_B = "PANEL_2X2"
PANELS_OTHER: List[str] = ["PANEL_1X4", "PANEL_2X4", "PANEL_4X4"]  # area arb absent
HEDGE_PANEL = 0.831                 # 2026-04-30 OLS refit (R^2=0.296) — was 0.47, LARGE change
PANELS_WINDOW = 1500

# --- Tier 4: NOPE ------------------------------------------------------------
OXYGEN_SHAKES = [
    "OXYGEN_SHAKE_MORNING_BREATH", "OXYGEN_SHAKE_EVENING_BREATH",
    "OXYGEN_SHAKE_MINT", "OXYGEN_SHAKE_CHOCOLATE", "OXYGEN_SHAKE_GARLIC",
]
# Skipped entirely — best pair's hedge ratio flipped sign across days
# (-1.48 -> +0.74 -> +0.96), no stable cointegration.

# --- Passive market making fallback ------------------------------------------
PASSIVE_MM_SIZE = 1                 # very small — toehold liquidity provision

# --- Universal MM layer (cover the 17 currently-untouched products) ----------
# Real platform PnL came in 10x below target — diagnosis is structural: many
# products receive zero orders. Bolt a passive-MM layer onto every product not
# already managed by a directional strategy. Size 1, inside-spread only, plus a
# microprice tilt to skew quote sizes toward the side the book is leaning.
UNIVERSAL_MM_SYMBOLS: List[str] = (
    list(OXYGEN_SHAKES)              # 5 — strategy was "skip", but passive MM is fine
    + [GALAXY_BLACK]                 # 1 — was avoided as trending
    + list(GALAXY_OTHER)             # 2 — DARK_MATTER, PLANETARY_RINGS
    + list(CHIPS_OTHER)              # 3 — CIRCLE, OVAL, TRIANGLE
    + list(TRANS_OTHER)              # 3 — SPACE_GRAY, ASTRO_BLACK, GRAPHITE_MIST
    + list(PANELS_OTHER)             # 3 — PANEL_1X4, 2X4, 4X4
    + [PANEL_LEG_A, PANEL_LEG_B]     # 2 — 2026-04-30 disabled panels pair (ADF non-stationary)
    + [ROBOT_DISHES]                 # 1 — 2026-04-30 disabled robots strategy (-706k in backtest)
    + list(ROBOTS_OTHER)             # 4 — was MM-only anyway
)                                    # total 24 covered products

# --- Microprice tilt (Tier A disruptive idea — applied to all MM quotes) -----
# imbalance = (bid_vol - ask_vol) / (bid_vol + ask_vol) ; range [-1, +1].
# When |imbalance| >= threshold, the book is leaning. Bias passive quote sizes
# toward the favored side and shrink the disfavored side.
MICROPRICE_TILT_THRESHOLD = 0.30
MICROPRICE_SKEW_SIZE = 1            # extra size added to favored side, removed from other

# --- market_trades flow signal (Tier A disruptive idea #1) -------------------
# Tracker stays on (cheap, populates history for future use), but the tilt
# weight is 0 for now — initial release with weight=0.5 coincided with a
# late-day PnL drawdown in user testing. Re-enable cautiously after we have
# a clearer read on whether the signal lags or leads.
MT_FLOW_WINDOW = 50                 # how many recent ticks to sum net flow over
MT_FLOW_HISTORY_MAX = 200           # cap per-product flow history length
MT_FLOW_WEIGHT = 0.0                # 2026-04-30 disabled (was 0.5)


# =============================================================================
#                                  LOGGING
# =============================================================================

def log(level: str, msg: str) -> None:
    """No-op unless LOG_LEVEL allows. Default OFF — silent in production."""
    if LOG_LEVEL == "OFF":
        return
    if LOG_LEVEL == "INFO" and level == "DEBUG":
        return
    # Stay light; platform captures stdout.
    print(f"[{level}] {msg}")


# =============================================================================
#                              HELPER FUNCTIONS
# =============================================================================

def best_bid_ask(order_depth: Optional[OrderDepth]) -> Tuple[Optional[int], Optional[int],
                                                             Optional[int], Optional[int]]:
    """Return (best_bid, best_bid_vol, best_ask, best_ask_vol).
    Volumes are returned as positive ints from the bid side and positive ints
    from the ask side (we negate the platform's negative ask sizes here).
    Returns Nones if a side is empty.
    """
    if order_depth is None:
        return None, None, None, None
    bb = bbv = ba = bav = None
    if order_depth.buy_orders:
        bb = max(order_depth.buy_orders.keys())
        bbv = order_depth.buy_orders[bb]
    if order_depth.sell_orders:
        ba = min(order_depth.sell_orders.keys())
        # platform stores ask volumes as negative — flip to positive
        bav = -order_depth.sell_orders[ba]
    return bb, bbv, ba, bav


def mid_price(order_depth: Optional[OrderDepth]) -> Optional[float]:
    """Top-of-book midpoint, or None if either side missing."""
    bb, _, ba, _ = best_bid_ask(order_depth)
    if bb is None or ba is None:
        return None
    return (bb + ba) / 2.0


def book_imbalance(order_depth: Optional[OrderDepth]) -> Optional[float]:
    """Top-of-book size imbalance in [-1, +1].

    +1 means all volume on bid side (likely upward pressure).
    -1 means all volume on ask side (likely downward pressure).
    Returns None if either side is missing or volumes are non-positive.
    """
    if order_depth is None:
        return None
    _bb, bbv, _ba, bav = best_bid_ask(order_depth)
    if bbv is None or bav is None:
        return None
    total = bbv + bav
    if total <= 0:
        return None
    return (bbv - bav) / total


def classify_market_trade_flow(trades: Optional[List[Trade]],
                               order_depth: Optional[OrderDepth]) -> int:
    """Compute net aggressor volume from a list of recent market trades.

    Sign convention: positive = aggressors hit the ASK (upward pressure);
    negative = aggressors hit the BID (downward pressure). Trades that print
    inside the spread are ignored as ambiguous.
    """
    if not trades or order_depth is None:
        return 0
    bb, _, ba, _ = best_bid_ask(order_depth)
    net = 0
    for t in trades:
        try:
            qty = abs(int(t.quantity))
            px = int(t.price)
        except Exception:
            continue
        if ba is not None and px >= ba:
            net += qty
        elif bb is not None and px <= bb:
            net -= qty
    return net


def market_flow_tilt(history: List[float], window: int = MT_FLOW_WINDOW) -> float:
    """Normalised net flow signal in [-1, +1] from recent per-tick net flows."""
    if not history:
        return 0.0
    recent = history[-window:]
    s = sum(recent)
    if s == 0:
        return 0.0
    avg_abs = sum(abs(x) for x in recent) / len(recent)
    if avg_abs <= 1e-6:
        return 0.0
    return max(-1.0, min(1.0, s / (avg_abs * len(recent))))


def clip_to_limit(qty: int, current_position: int, cap: int = POSITION_LIMIT) -> int:
    """Clip a desired order quantity so post-trade |position| <= cap.

    qty positive = BUY, negative = SELL.
    Returns the clipped quantity (still signed). 0 if no headroom.
    """
    if qty == 0:
        return 0
    if qty > 0:
        headroom = cap - current_position
        if headroom <= 0:
            return 0
        return min(qty, headroom)
    else:  # qty < 0
        headroom = cap + current_position  # how much we can sell before -cap
        if headroom <= 0:
            return 0
        return -min(-qty, headroom)


def push_history(history: List[float], value: float, max_len: int = DEFAULT_MAX_HISTORY) -> None:
    """Append, then trim from the front if exceeding max_len."""
    history.append(float(value))
    overflow = len(history) - max_len
    if overflow > 0:
        del history[:overflow]


def rolling_mean_std(history: List[float], window: int) -> Tuple[Optional[float], Optional[float]]:
    """Mean and (sample) std over the last `window` points. Returns (None, None) if too short."""
    if len(history) < max(window, 2):
        return None, None
    window_data = history[-window:]
    mu = sum(window_data) / len(window_data)
    # variance via two-pass for numerical stability
    var = sum((x - mu) ** 2 for x in window_data) / (len(window_data) - 1)
    sd = math.sqrt(var) if var > 0 else 0.0
    return mu, sd


def safe_z(x: float, mu: Optional[float], sd: Optional[float]) -> Optional[float]:
    """Z-score with guards against None / zero std."""
    if mu is None or sd is None or sd <= 1e-9:
        return None
    return (x - mu) / sd


def linear_size(z: float, max_size: int) -> int:
    """Linear sizing: at z=2.0 use full max_size; clamp to +-max_size.
    Sign of returned size is opposite to sign of z (mean reversion)."""
    if z is None:
        return 0
    raw = -1.0 * (max(-1.0, min(1.0, z / 2.0))) * max_size
    return int(round(raw))


def merge_orders(target: Dict[str, List[Order]], extra: Dict[str, List[Order]]) -> None:
    """Merge per-symbol order lists in-place."""
    for sym, lst in extra.items():
        if not lst:
            continue
        target.setdefault(sym, []).extend(lst)


# =============================================================================
#                              PERSISTENT STATE
# =============================================================================
# Persisted via state.traderData (JSON-serialised string).
# Schema:
# {
#   "history": { "<key>": [floats...] },     # rolling histories per signal
#   "v": 1                                   # version tag for migrations
# }


def load_state(traderData: str) -> dict:
    """Robust load with fallbacks. Never raises."""
    if not traderData:
        return {"history": {}, "v": 1}
    try:
        st = json.loads(traderData)
        if not isinstance(st, dict):
            return {"history": {}, "v": 1}
        st.setdefault("history", {})
        st.setdefault("v", 1)
        return st
    except Exception:
        return {"history": {}, "v": 1}


def dump_state(state_dict: dict) -> str:
    """Compact JSON. Never raises — returns empty string on failure."""
    try:
        return json.dumps(state_dict, separators=(",", ":"))
    except Exception:
        return ""


def get_history(state: dict, key: str) -> List[float]:
    h = state["history"].setdefault(key, [])
    return h


# =============================================================================
#                                   TRADER
# =============================================================================

class Trader:
    """Round-5 baseline trader. Combines per-category sub-strategies."""

    # ---------------------------------------------------------------------
    #  Top-level dispatch
    # ---------------------------------------------------------------------
    def run(self, state: TradingState) -> Tuple[Dict[str, List[Order]], int, str]:
        result: Dict[str, List[Order]] = {}
        conversions = 0

        # Load persistent state (rolling histories etc.).
        persistent = load_state(state.traderData or "")

        # Update per-product market_trades flow histories (Tier A disruptive #1).
        # Cheap pass — feeds the MM tilt logic in _run_passive_mm.
        try:
            self._track_market_flow(state, persistent)
        except Exception as e:
            log("INFO", f"market-flow track failed: {e}")

        try:
            merge_orders(result, self._run_pebbles(state, persistent))
        except Exception as e:
            log("INFO", f"pebbles failed: {e}")

        try:
            merge_orders(result, self._run_classics(state, persistent))
        except Exception as e:
            log("INFO", f"classics failed: {e}")

        try:
            merge_orders(result, self._run_berries(state, persistent))
        except Exception as e:
            log("INFO", f"berries failed: {e}")

        try:
            merge_orders(result, self._run_passive_mm(state, SNACK_FALLBACK_MM, persistent))
        except Exception as e:
            log("INFO", f"snacks-mm failed: {e}")

        try:
            merge_orders(result, self._run_sleep_pair_polyester_cotton(state, persistent))
        except Exception as e:
            log("INFO", f"sleep-pc failed: {e}")

        try:
            merge_orders(result, self._run_sleep_pair_lamb_nylon(state, persistent))
        except Exception as e:
            log("INFO", f"sleep-ln failed: {e}")

        try:
            merge_orders(result, self._run_passive_mm(state, [SLEEP_SUEDE], persistent))
        except Exception as e:
            log("INFO", f"sleep-suede-mm failed: {e}")

        # Robots strategy DISABLED 2026-04-30 — backtest showed -706k (single biggest hole).
        # All 5 robot products are now covered by the universal MM layer below instead.

        try:
            merge_orders(result, self._run_visors(state, persistent))
        except Exception as e:
            log("INFO", f"visors failed: {e}")

        try:
            merge_orders(result, self._run_passive_mm(state, VISORS_OTHER, persistent))
        except Exception as e:
            log("INFO", f"visors-mm failed: {e}")

        # Bronze pairs — small caps
        try:
            merge_orders(result, self._run_bronze_pair(
                state, persistent,
                key="galaxy",
                a=GALAXY_FLAMES, b=GALAXY_WINDS,
                hedge=HEDGE_FLAMES_WINDS,    # spread = A + h*B
                spread_sign_b=+1.0,
                window=GALAXY_WINDOW))
        except Exception as e:
            log("INFO", f"galaxy failed: {e}")

        try:
            merge_orders(result, self._run_bronze_pair(
                state, persistent,
                key="microchips",
                a=CHIP_SQUARE, b=CHIP_RECT,
                hedge=HEDGE_SQUARE_RECT,
                spread_sign_b=+1.0,
                window=CHIPS_WINDOW))
        except Exception as e:
            log("INFO", f"microchips failed: {e}")

        try:
            merge_orders(result, self._run_bronze_pair(
                state, persistent,
                key="translators",
                a=TRANS_CHARCOAL, b=TRANS_VOIDBLUE,
                hedge=HEDGE_CHARCOAL_VOIDBLUE,
                spread_sign_b=-1.0,             # spread = A - h*B
                window=TRANS_WINDOW))
        except Exception as e:
            log("INFO", f"translators failed: {e}")

        # Panels pair DISABLED 2026-04-30 — ADF on residual = PANEL_1X2 + 0.83 *
        # PANEL_2X2 returned p=0.099 over the 3-day pooled sample (vs <0.05 for
        # all other pairs incl. weak-R² FLAMES/WINDS at p=0.021). Non-stationary
        # residual means there's no mean to revert to; the strategy was paying
        # spread on a drifting series. The two legs are now routed through
        # universal MM instead.

        # Universal passive-MM layer — covers the 17 products none of the
        # directional strategies above touch (oxygen shakes + bronze "OTHER"
        # legs + GALAXY_BLACK). Microprice-tilted, size 1-2 per side.
        try:
            merge_orders(result, self._run_passive_mm(state, UNIVERSAL_MM_SYMBOLS, persistent))
        except Exception as e:
            log("INFO", f"universal-mm failed: {e}")

        # Defensive global pass: clip all orders to position limits one more time.
        result = self._final_clip_pass(result, state)

        traderData = dump_state(persistent)
        return result, conversions, traderData

    # ---------------------------------------------------------------------
    #  Final defensive clip
    # ---------------------------------------------------------------------
    def _final_clip_pass(self,
                         orders: Dict[str, List[Order]],
                         state: TradingState) -> Dict[str, List[Order]]:
        """Walk every product's orders. Sum buys and sells separately and
        clip so that the worst-case post-fill position respects POSITION_LIMIT.
        We assume the same cap unless the symbol is bronze (cap=3)."""
        cleaned: Dict[str, List[Order]] = {}
        bronze_syms = {GALAXY_FLAMES, GALAXY_WINDS,
                       CHIP_SQUARE, CHIP_RECT,
                       TRANS_CHARCOAL, TRANS_VOIDBLUE,
                       PANEL_LEG_A, PANEL_LEG_B}

        for sym, lst in orders.items():
            if not lst:
                continue
            cap = BRONZE_POSITION_CAP if sym in bronze_syms else POSITION_LIMIT
            pos = state.position.get(sym, 0)
            running_buy = 0
            running_sell = 0
            kept: List[Order] = []
            for o in lst:
                if o.quantity == 0:
                    continue
                if o.quantity > 0:
                    # Worst-case after this order: pos + running_buy + o.quantity
                    headroom = cap - (pos + running_buy)
                    q = min(o.quantity, headroom)
                    if q > 0:
                        running_buy += q
                        kept.append(Order(sym, o.price, q))
                else:
                    headroom = cap + (pos - running_sell)
                    q = min(-o.quantity, headroom)
                    if q > 0:
                        running_sell += q
                        kept.append(Order(sym, o.price, -q))
            if kept:
                cleaned[sym] = kept
        return cleaned

    # =====================================================================
    #                              PEBBLES
    # =====================================================================
    def _run_pebbles(self, state: TradingState, persistent: dict) -> Dict[str, List[Order]]:
        """Residual-conditioned passive market making per leg (rolled back from
        the single-leg cross-spread variant on 2026-04-30 — that fired with
        full-spread cost and rolling-mean lag, hurting late-day PnL).

        Strategy:
          resid = (mid_XS + mid_S + mid_M + mid_L + mid_XL) - PEBBLES_TARGET_SUM
          - |resid| <= PEBBLES_RESID_TRIGGER -> no directional quotes.
          - resid > trigger  -> basket overpriced; mids about to drop. Per leg,
            join the best ask passively (post sell at ask_1).
          - resid < -trigger -> basket underpriced. Per leg, join the best bid
            passively (post buy at bid_1).
          PLUS inventory unwind: pos>0 -> small passive sell at ask_1;
            pos<0 -> small passive buy at bid_1. Quotes merge per side.
        """
        out: Dict[str, List[Order]] = {}

        depths = {p: state.order_depths.get(p) for p in PEBBLES}
        positions = {p: state.position.get(p, 0) for p in PEBBLES}

        bb: Dict[str, Optional[int]] = {}
        ba: Dict[str, Optional[int]] = {}
        mids: Dict[str, Optional[float]] = {}
        for p in PEBBLES:
            b1, _, a1, _ = best_bid_ask(depths[p])
            bb[p], ba[p] = b1, a1
            mids[p] = mid_price(depths[p])

        if any(mids[p] is None for p in PEBBLES):
            return out

        resid = sum(mids[p] for p in PEBBLES) - PEBBLES_TARGET_SUM  # type: ignore

        buy_qty: Dict[str, int] = {p: 0 for p in PEBBLES}
        sell_qty: Dict[str, int] = {p: 0 for p in PEBBLES}

        if resid > PEBBLES_RESID_TRIGGER:
            for p in PEBBLES:
                if ba[p] is None:
                    continue
                headroom = POSITION_LIMIT + positions[p]
                if headroom <= 0:
                    continue
                sell_qty[p] = max(sell_qty[p], min(PEBBLES_PASSIVE_SIZE, headroom))
        elif resid < -PEBBLES_RESID_TRIGGER:
            for p in PEBBLES:
                if bb[p] is None:
                    continue
                headroom = POSITION_LIMIT - positions[p]
                if headroom <= 0:
                    continue
                buy_qty[p] = max(buy_qty[p], min(PEBBLES_PASSIVE_SIZE, headroom))

        for p in PEBBLES:
            pos = positions[p]
            if pos > 0 and ba[p] is not None:
                unwind = min(PEBBLES_UNWIND_SIZE, pos)
                if unwind > 0:
                    sell_qty[p] = max(sell_qty[p], unwind)
            elif pos < 0 and bb[p] is not None:
                unwind = min(PEBBLES_UNWIND_SIZE, -pos)
                if unwind > 0:
                    buy_qty[p] = max(buy_qty[p], unwind)

        for p in PEBBLES:
            if buy_qty[p] > 0 and bb[p] is not None:
                qty = clip_to_limit(buy_qty[p], positions[p])
                if qty > 0:
                    out.setdefault(p, []).append(Order(p, bb[p], qty))
            if sell_qty[p] > 0 and ba[p] is not None:
                qty = clip_to_limit(-sell_qty[p], positions[p])
                if qty < 0:
                    out.setdefault(p, []).append(Order(p, ba[p], qty))

        return out

    # =====================================================================
    #                          SNACK_PACKS / CLASSICS
    # =====================================================================
    def _run_classics(self, state: TradingState, persistent: dict) -> Dict[str, List[Order]]:
        return self._run_two_leg_basket(
            state, persistent,
            key="classics",
            legs=CLASSICS_LEGS,
            window=CLASSICS_WINDOW,
            z_enter=SNACK_Z_ENTER,
            z_exit=SNACK_Z_EXIT,
            z_stop=SNACK_Z_STOP,
            max_leg_size=SNACK_MAX_LEG_SIZE,
        )

    def _run_berries(self, state: TradingState, persistent: dict) -> Dict[str, List[Order]]:
        return self._run_two_leg_basket(
            state, persistent,
            key="berries",
            legs=BERRIES_LEGS,
            window=BERRIES_WINDOW,
            z_enter=SNACK_Z_ENTER,
            z_exit=SNACK_Z_EXIT,
            z_stop=SNACK_Z_STOP,
            max_leg_size=SNACK_MAX_LEG_SIZE,
        )

    def _run_two_leg_basket(self,
                            state: TradingState,
                            persistent: dict,
                            key: str,
                            legs: List[str],
                            window: int,
                            z_enter: float,
                            z_exit: float,
                            z_stop: float,
                            max_leg_size: int) -> Dict[str, List[Order]]:
        """Generic 2-leg sum basket: signal = mid_a + mid_b. Trade BOTH legs same direction.

        At z=+2: basket high -> sell legs (sell each).
        At z=-2: basket low  -> buy legs (buy each).
        """
        out: Dict[str, List[Order]] = {}
        depths = [state.order_depths.get(s) for s in legs]
        if any(d is None for d in depths):
            return out
        mids = [mid_price(d) for d in depths]
        if any(m is None for m in mids):
            return out

        signal = sum(mids)  # type: ignore
        hist = get_history(persistent, f"{key}_signal")
        push_history(hist, signal, max_len=max(DEFAULT_MAX_HISTORY, window + 50))
        mu, sd = rolling_mean_std(hist, window)
        z = safe_z(signal, mu, sd)
        if z is None:
            return out

        # --- Decide target leg position ---
        positions = [state.position.get(s, 0) for s in legs]

        # Stop-out: |z| > z_stop -> flatten; |z| < z_exit -> exit; |z| >= z_enter -> enter;
        # in between -> hold (early return so we don't drag legs around).
        if abs(z) > z_stop or abs(z) < z_exit:
            target_each = 0
        elif abs(z) >= z_enter:
            target_each = linear_size(z, max_leg_size)
        else:
            return out  # hold band — no orders

        # Build orders to move each leg's position toward target_each.
        for s, d, pos in zip(legs, depths, positions):
            delta = target_each - pos
            if delta == 0:
                continue
            bb, bbv, ba, bav = best_bid_ask(d)
            if delta > 0:
                # Need to BUY — cross the ask
                if ba is None:
                    continue
                qty = clip_to_limit(delta, pos)
                if qty > 0:
                    out.setdefault(s, []).append(Order(s, ba, qty))
            else:
                if bb is None:
                    continue
                qty = clip_to_limit(delta, pos)
                if qty < 0:
                    out.setdefault(s, []).append(Order(s, bb, qty))
        return out

    # =====================================================================
    #                            SLEEP PAIRS
    # =====================================================================
    def _run_sleep_pair_polyester_cotton(self,
                                         state: TradingState,
                                         persistent: dict) -> Dict[str, List[Order]]:
        return self._run_two_leg_pair(
            state, persistent,
            key="sleep_pc",
            a=SLEEP_POLY, b=SLEEP_COTTON,
            hedge=HEDGE_POLY_COTTON,
            spread_sign_b=-1.0,        # spread = A - hedge*B
            window=SLEEP_WINDOW,
            z_enter=SLEEP_Z_ENTER,
            z_exit=SLEEP_Z_EXIT,
            z_stop=SLEEP_Z_STOP,
            cap=SLEEP_MAX_LEG_SIZE,
        )

    def _run_sleep_pair_lamb_nylon(self,
                                   state: TradingState,
                                   persistent: dict) -> Dict[str, List[Order]]:
        return self._run_two_leg_pair(
            state, persistent,
            key="sleep_ln",
            a=SLEEP_LAMB, b=SLEEP_NYLON,
            hedge=HEDGE_LAMB_NYLON,
            spread_sign_b=-1.0,
            window=SLEEP_WINDOW,
            z_enter=SLEEP_Z_ENTER,
            z_exit=SLEEP_Z_EXIT,
            z_stop=SLEEP_Z_STOP,
            cap=SLEEP_MAX_LEG_SIZE,
        )

    def _run_two_leg_pair(self,
                          state: TradingState,
                          persistent: dict,
                          key: str,
                          a: str,
                          b: str,
                          hedge: float,
                          spread_sign_b: float,    # +1 means spread=A+h*B, -1 means spread=A-h*B
                          window: int,
                          z_enter: float,
                          z_exit: float,
                          z_stop: float,
                          cap: int) -> Dict[str, List[Order]]:
        """Pair trade with hedge ratio.

        spread = mid_A + spread_sign_b * hedge * mid_B
        At z=+2: short the spread.  A "unit short of spread" position is
        pos_A = -1, pos_B = -spread_sign_b * hedge  (so portfolio value = -spread).
        With target_A = linear_size(z, cap) (already signed, negative when z>0),
        the matched hedge leg is target_B = spread_sign_b * hedge * target_A.
        Hold-band -> early return (don't drag legs around when no signal).
        """
        out: Dict[str, List[Order]] = {}
        da = state.order_depths.get(a)
        db = state.order_depths.get(b)
        if da is None or db is None:
            return out
        ma = mid_price(da)
        mb = mid_price(db)
        if ma is None or mb is None:
            return out

        spread = ma + spread_sign_b * hedge * mb
        hist = get_history(persistent, f"{key}_spread")
        push_history(hist, spread, max_len=max(DEFAULT_MAX_HISTORY, window + 50))
        mu, sd = rolling_mean_std(hist, window)
        z = safe_z(spread, mu, sd)
        if z is None:
            return out

        pos_a = state.position.get(a, 0)
        pos_b = state.position.get(b, 0)

        if abs(z) > z_stop or abs(z) < z_exit:
            target_a = 0
        elif abs(z) >= z_enter:
            target_a = linear_size(z, cap)
        else:
            return out  # hold band — leave both legs alone

        # Hedge leg target. spread = A + s*h*B; "unit short of spread" needs
        # pos_A = -1, pos_B = -s*h, so for any signed target_A the matched hedge
        # is target_B = s*h*target_A.
        target_b = int(round(spread_sign_b * hedge * target_a))
        target_b = max(-cap, min(cap, target_b))

        self._send_to_target(out, a, da, pos_a, target_a)
        self._send_to_target(out, b, db, pos_b, target_b)
        return out

    def _send_to_target(self,
                        out: Dict[str, List[Order]],
                        sym: str,
                        depth: OrderDepth,
                        current: int,
                        target: int,
                        cap: int = POSITION_LIMIT) -> None:
        """Cross top-of-book to move position toward target. Clips to cap."""
        delta = target - current
        if delta == 0:
            return
        bb, bbv, ba, bav = best_bid_ask(depth)
        if delta > 0:
            if ba is None:
                return
            qty = clip_to_limit(delta, current, cap=cap)
            if qty > 0:
                out.setdefault(sym, []).append(Order(sym, ba, qty))
        else:
            if bb is None:
                return
            qty = clip_to_limit(delta, current, cap=cap)
            if qty < 0:
                out.setdefault(sym, []).append(Order(sym, bb, qty))

    # =====================================================================
    #                                ROBOTS
    # =====================================================================
    def _run_robots(self, state: TradingState, persistent: dict) -> Dict[str, List[Order]]:
        """DISHES vs equal-weight basket of (VACUUMING, MOPPING, LAUNDRY, IRONING).

        residual = mid_DISHES - mean(mid of other 4)
        z>0 -> DISHES rich -> short DISHES, long 1/4 of pos in each other.
        """
        out: Dict[str, List[Order]] = {}
        dd = state.order_depths.get(ROBOT_DISHES)
        others_depths = {s: state.order_depths.get(s) for s in ROBOTS_OTHER}
        if dd is None or any(v is None for v in others_depths.values()):
            return out
        md = mid_price(dd)
        others_mids = {s: mid_price(d) for s, d in others_depths.items()}
        if md is None or any(m is None for m in others_mids.values()):
            return out

        basket_avg = sum(others_mids.values()) / len(others_mids)  # type: ignore
        residual = md - basket_avg
        hist = get_history(persistent, "robots_resid")
        push_history(hist, residual, max_len=max(DEFAULT_MAX_HISTORY, ROBOTS_WINDOW + 50))
        mu, sd = rolling_mean_std(hist, ROBOTS_WINDOW)
        z = safe_z(residual, mu, sd)
        if z is None:
            return out

        pos_d = state.position.get(ROBOT_DISHES, 0)

        if abs(z) > ROBOTS_Z_STOP or abs(z) < ROBOTS_Z_EXIT:
            target_d = 0
        elif abs(z) >= ROBOTS_Z_ENTER:
            target_d = linear_size(z, ROBOTS_MAX_LEG_SIZE)
        else:
            target_d = pos_d

        # Each "other" leg gets -1/4 of DISHES position (rounded).
        per_other = int(round(-target_d / 4.0))
        per_other = max(-ROBOTS_MAX_LEG_SIZE, min(ROBOTS_MAX_LEG_SIZE, per_other))

        self._send_to_target(out, ROBOT_DISHES, dd, pos_d, target_d)
        for s in ROBOTS_OTHER:
            d = others_depths[s]
            if d is None:
                continue
            cur = state.position.get(s, 0)
            self._send_to_target(out, s, d, cur, per_other)
        return out

    # =====================================================================
    #                               UV_VISORS
    # =====================================================================
    def _run_visors(self, state: TradingState, persistent: dict) -> Dict[str, List[Order]]:
        """Pair: spread = mid_AMBER + 0.53 * mid_MAGENTA."""
        return self._run_two_leg_pair(
            state, persistent,
            key="visors_am",
            a=VISOR_AMBER, b=VISOR_MAGENTA,
            hedge=HEDGE_AMBER_MAGENTA,
            spread_sign_b=+1.0,
            window=VISORS_WINDOW,
            z_enter=VISORS_Z_ENTER,
            z_exit=VISORS_Z_EXIT,
            z_stop=VISORS_Z_STOP,
            cap=VISORS_MAX_LEG_SIZE,
        )

    # =====================================================================
    #                          BRONZE PAIR (small size)
    # =====================================================================
    def _run_bronze_pair(self,
                         state: TradingState,
                         persistent: dict,
                         key: str,
                         a: str,
                         b: str,
                         hedge: float,
                         spread_sign_b: float,
                         window: int) -> Dict[str, List[Order]]:
        """Bronze-tier pair trade with conservative thresholds and |pos|<=3."""
        return self._run_two_leg_pair(
            state, persistent,
            key=key,
            a=a, b=b,
            hedge=hedge,
            spread_sign_b=spread_sign_b,
            window=window,
            z_enter=BRONZE_Z_ENTER,
            z_exit=BRONZE_Z_EXIT,
            z_stop=BRONZE_Z_STOP,
            cap=BRONZE_POSITION_CAP,
        )

    # =====================================================================
    #                       PASSIVE MARKET MAKING FALLBACK
    # =====================================================================
    def _track_market_flow(self,
                           state: TradingState,
                           persistent: dict) -> None:
        """Append this tick's net aggressor volume per symbol to history."""
        market_trades = state.market_trades or {}
        for sym in state.order_depths.keys():
            trades = market_trades.get(sym)
            depth = state.order_depths.get(sym)
            net = classify_market_trade_flow(trades, depth)
            hist = get_history(persistent, f"mt_flow_{sym}")
            push_history(hist, float(net), max_len=MT_FLOW_HISTORY_MAX)

    def _run_passive_mm(self,
                        state: TradingState,
                        symbols: Iterable[str],
                        persistent: Optional[dict] = None) -> Dict[str, List[Order]]:
        """Post inside-spread bid/ask of size PASSIVE_MM_SIZE for each symbol,
        with combined microprice + market_trades flow tilt that skews sizes
        toward whichever side the book is leaning. Skips a side if posting it
        would breach the cap.
        """
        out: Dict[str, List[Order]] = {}
        for sym in symbols:
            depth = state.order_depths.get(sym)
            if depth is None:
                continue
            bb, _, ba, _ = best_bid_ask(depth)
            if bb is None or ba is None:
                continue
            if ba - bb < 2:
                # nothing to make inside; skip rather than join
                continue
            pos = state.position.get(sym, 0)
            bid_px = bb + 1
            ask_px = ba - 1

            bid_size = PASSIVE_MM_SIZE
            ask_size = PASSIVE_MM_SIZE

            imb = book_imbalance(depth) or 0.0
            flow_tilt = 0.0
            if persistent is not None:
                flow_hist = persistent.get("history", {}).get(f"mt_flow_{sym}", [])
                flow_tilt = market_flow_tilt(flow_hist)
            combined = imb + MT_FLOW_WEIGHT * flow_tilt

            if combined >= MICROPRICE_TILT_THRESHOLD:
                # upward pressure — keep bid, shrink ask (don't sell into a rally)
                bid_size += MICROPRICE_SKEW_SIZE
                ask_size = max(0, ask_size - MICROPRICE_SKEW_SIZE)
            elif combined <= -MICROPRICE_TILT_THRESHOLD:
                ask_size += MICROPRICE_SKEW_SIZE
                bid_size = max(0, bid_size - MICROPRICE_SKEW_SIZE)

            bq = clip_to_limit(bid_size, pos) if bid_size > 0 else 0
            aq = clip_to_limit(-ask_size, pos) if ask_size > 0 else 0
            if bq > 0:
                out.setdefault(sym, []).append(Order(sym, bid_px, bq))
            if aq < 0:
                out.setdefault(sym, []).append(Order(sym, ask_px, aq))
        return out


# =============================================================================
#                                    MAIN
# =============================================================================

if __name__ == "__main__":
    # Instantiate a Trader so any naming/syntax error trips here.
    _t = Trader()
    print("loaded ok")
