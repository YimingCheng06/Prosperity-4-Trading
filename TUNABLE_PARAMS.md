# Tunable Hyperparameters — `trader_baseline.py`

This document enumerates every tunable hyperparameter in `trader_baseline.py`,
organised by category. Each entry lists:

- **Constant** — name exactly as in code.
- **Current** — the literal default in the source.
- **Meaning** — one-line physical interpretation.
- **Grid** — suggested sweep range for a hyperparameter search.
- **Type** — `H` (true hyperparameter, grid-search) or `S` (statistical
  estimate; refit per day via OLS / rolling regression and **do not** include
  in the grid).

Default grid templates (used wherever a constant fits the pattern):

```
z_enter   : {1.0, 1.5, 2.0, 2.5, 3.0}    (5 levels)
z_exit    : {0.0, 0.2, 0.4}              (3 levels)
z_stop    : {3.0, 3.5, 4.0, 5.0}         (4 levels)
window    : {250, 500, 1000, 1500, 2000} (5 levels)
```

Size caps are NOT swept — they are constrained by the platform position limit
(`POSITION_LIMIT = 10` for Gold/Silver, `BRONZE_POSITION_CAP = 3` for Bronze).

---

## 1. Globals

| Constant              | Current | Meaning                                                | Grid           | Type |
|-----------------------|---------|--------------------------------------------------------|----------------|------|
| `POSITION_LIMIT`      | 10      | Hard exchange limit per product. Fixed by platform.    | (fixed)        | —    |
| `LOG_LEVEL`           | "OFF"   | Verbosity. Not tunable for PnL.                        | (fixed)        | —    |
| `DEFAULT_MAX_HISTORY` | 2000    | Rolling deque max length per signal. Minor PnL effect. | (fixed)        | —    |
| `PASSIVE_MM_SIZE`     | 1       | Size posted by fallback inside-spread MM on quiet syms.| {1, 2, 3}      | H    |

Only `PASSIVE_MM_SIZE` is interesting to sweep here (3 levels).

---

## 2. Pebbles

Strategy: residual-conditioned passive MM per leg + inventory unwind.

| Constant                  | Current | Meaning                                                            | Grid                     | Type |
|---------------------------|---------|--------------------------------------------------------------------|--------------------------|------|
| `PEBBLES_TARGET_SUM`      | 50000   | Mechanical conservation constant. Not tunable.                     | (fixed)                  | —    |
| `THRESHOLD_PEBBLES_TAKE`  | 9999    | Disabled-take sentinel. Deep-dive proved take-arb unprofitable.    | (fixed at 9999)          | —    |
| `PEBBLES_RESID_TRIGGER`   | 1       | Residual abs-threshold to fire directional quotes.                 | {0, 1, 2, 3, 5, 10}      | H    |
| `PEBBLES_PASSIVE_SIZE`    | 2       | Size of residual-conditioned passive quote per leg.                | {1, 2, 3, 4}             | H    |
| `PEBBLES_PASSIVE_TTL`     | 5       | Documentation only — platform cancels passive orders each tick.    | (fixed)                  | —    |
| `PEBBLES_UNWIND_SIZE`     | 2       | Size of inventory-unwind passive quote when `|pos|>0`.             | {1, 2, 3}                | H    |

Pebbles grid size: 6 * 4 * 3 = **72**.

---

## 3. Snack Packs (Classics + Berries)

Strategy: 2-leg sum-basket z-score reversion (CHOC+VAN, STRAW+RASP).
Pistachio uses fallback passive MM only.

| Constant            | Current        | Meaning                                              | Grid                     | Type |
|---------------------|----------------|------------------------------------------------------|--------------------------|------|
| `CLASSICS_WINDOW`   | 500            | Rolling window for CHOC+VAN signal.                  | {250, 500, 1000, 1500}   | H    |
| `BERRIES_WINDOW`    | 500            | Rolling window for STRAW+RASP signal.                | {250, 500, 1000, 1500}   | H    |
| `SNACK_Z_ENTER`     | 1.5            | Entry threshold (shared across both pairs).          | {1.0, 1.5, 2.0, 2.5}     | H    |
| `SNACK_Z_EXIT`      | 0.3            | Exit threshold.                                      | {0.0, 0.2, 0.4}          | H    |
| `SNACK_Z_STOP`      | 3.0            | Stop-out threshold.                                  | {3.0, 3.5, 4.0, 5.0}     | H    |
| `SNACK_MAX_LEG_SIZE`| 10             | Max position per leg (= POSITION_LIMIT).             | (fixed)                  | —    |

Snacks grid size (treating CLASSICS_WINDOW and BERRIES_WINDOW as a single
"window" for shared-z legs, since z thresholds are shared): 4 * 4 * 3 * 4 =
**192**. If you sweep windows independently per pair, multiply by 4 again.

---

## 4. Sleep Pods (POLY/COTTON + LAMB/NYLON)

Strategy: two pair-spreads, hedged. Suede uses fallback passive MM only.

| Constant             | Current | Meaning                                                | Grid                     | Type |
|----------------------|---------|--------------------------------------------------------|--------------------------|------|
| `SLEEP_WINDOW`       | 1000    | Rolling window for both sleep-pair spreads.            | {500, 1000, 1500, 2000}  | H    |
| `SLEEP_Z_ENTER`      | 2.0     | Entry threshold (shared).                              | {1.0, 1.5, 2.0, 2.5, 3.0}| H    |
| `SLEEP_Z_EXIT`       | 0.3     | Exit threshold.                                        | {0.0, 0.2, 0.4}          | H    |
| `SLEEP_Z_STOP`       | 3.5     | Stop-out threshold.                                    | {3.0, 3.5, 4.0, 5.0}     | H    |
| `SLEEP_MAX_LEG_SIZE` | 10      | Max leg size.                                          | (fixed)                  | —    |

Sleep grid size: 4 * 5 * 3 * 4 = **240**.

---

## 5. Robots

Strategy: DISHES vs equal-weight basket of (VACUUMING, MOPPING, LAUNDRY,
IRONING). Other legs use fallback passive MM as well.

| Constant              | Current | Meaning                                                | Grid                     | Type |
|-----------------------|---------|--------------------------------------------------------|--------------------------|------|
| `ROBOTS_WINDOW`       | 1000    | Rolling window for residual.                           | {500, 1000, 1500, 2000}  | H    |
| `ROBOTS_Z_ENTER`      | 2.0     | Entry threshold.                                       | {1.0, 1.5, 2.0, 2.5, 3.0}| H    |
| `ROBOTS_Z_EXIT`       | 0.3     | Exit threshold.                                        | {0.0, 0.2, 0.4}          | H    |
| `ROBOTS_Z_STOP`       | 3.5     | Stop-out threshold.                                    | {3.0, 3.5, 4.0, 5.0}     | H    |
| `ROBOTS_MAX_LEG_SIZE` | 10      | Max position per leg.                                  | (fixed)                  | —    |

Robots grid size: 4 * 5 * 3 * 4 = **240**.

---

## 6. UV Visors

Strategy: AMBER + 0.53*MAGENTA pair. Other visor colours use fallback MM.

| Constant              | Current | Meaning                                                | Grid                     | Type |
|-----------------------|---------|--------------------------------------------------------|--------------------------|------|
| `VISORS_WINDOW`       | 1000    | Rolling window for spread.                             | {500, 1000, 1500, 2000}  | H    |
| `VISORS_Z_ENTER`      | 2.0     | Entry threshold.                                       | {1.0, 1.5, 2.0, 2.5, 3.0}| H    |
| `VISORS_Z_EXIT`       | 0.3     | Exit threshold.                                        | {0.0, 0.2, 0.4}          | H    |
| `VISORS_Z_STOP`       | 3.5     | Stop-out threshold.                                    | {3.0, 3.5, 4.0, 5.0}     | H    |
| `VISORS_MAX_LEG_SIZE` | 10      | Max position per leg.                                  | (fixed)                  | —    |

Visors grid size: 4 * 5 * 3 * 4 = **240**.

---

## 7. Bronze Tier (Galaxy / Microchips / Translators / Panels)

All four bronze pairs share `BRONZE_Z_*` thresholds and use
`BRONZE_POSITION_CAP = 3`. Each pair has its own rolling window.

| Constant               | Current | Meaning                                                  | Grid                     | Type |
|------------------------|---------|----------------------------------------------------------|--------------------------|------|
| `BRONZE_POSITION_CAP`  | 3       | Per-leg position cap for all bronze products. Fixed.     | (fixed)                  | —    |
| `BRONZE_Z_ENTER`       | 2.5     | Shared entry threshold (more conservative).              | {2.0, 2.5, 3.0, 3.5}     | H    |
| `BRONZE_Z_EXIT`        | 0.3     | Shared exit threshold.                                   | {0.0, 0.2, 0.4}          | H    |
| `BRONZE_Z_STOP`        | 4.0     | Shared stop-out threshold.                               | {3.5, 4.0, 4.5, 5.0}     | H    |
| `GALAXY_WINDOW`        | 1500    | Rolling window for FLAMES+0.28*WINDS spread.             | {500, 1000, 1500, 2000}  | H    |
| `CHIPS_WINDOW`         | 1500    | Rolling window for SQUARE+2.15*RECTANGLE spread.         | {500, 1000, 1500, 2000}  | H    |
| `TRANS_WINDOW`         | 1500    | Rolling window for CHARCOAL-0.287*VOIDBLUE spread.       | {500, 1000, 1500, 2000}  | H    |
| `PANELS_WINDOW`        | 1500    | Rolling window for PANEL_1X2+0.47*PANEL_2X2 spread.      | {500, 1000, 1500, 2000}  | H    |

Bronze grid size:
- shared z-knobs: 4 * 3 * 4 = 48
- 4 windows independent: 4^4 = 256
- naive product: 48 * 256 = **12288** (too large).
- Recommended: sweep z-knobs jointly across all bronze (48), and each window
  independently per pair (4 each). Scoring per pair: 48 * 4 = 192 evals;
  total bronze evals = 4 * 192 = **768** (since pairs are independent in PnL).

---

## 8. Hedge Ratios — RE-FIT VIA OLS, do not grid-search

These are statistical estimates of cointegrating vectors. They should be
re-estimated each day via OLS (or a rolling-window regression) on the most
recent in-sample data. Including them in a grid is a category error and
will overfit.

| Constant                   | Current | Spread definition                                                   | Type |
|----------------------------|---------|---------------------------------------------------------------------|------|
| `HEDGE_POLY_COTTON`        | 0.964   | spread = mid_POLY - 0.964 * mid_COTTON                              | S    |
| `HEDGE_LAMB_NYLON`         | 0.401   | spread = mid_LAMB - 0.401 * mid_NYLON                               | S    |
| `HEDGE_AMBER_MAGENTA`      | 0.53    | spread = mid_AMBER + 0.53 * mid_MAGENTA                             | S    |
| `HEDGE_FLAMES_WINDS`       | 0.28    | spread = mid_FLAMES + 0.28 * mid_WINDS                              | S    |
| `HEDGE_SQUARE_RECT`        | 2.15    | spread = mid_SQUARE + 2.15 * mid_RECTANGLE                          | S    |
| `HEDGE_CHARCOAL_VOIDBLUE`  | 0.287   | spread = mid_CHARCOAL - 0.287 * mid_VOIDBLUE                        | S    |
| `HEDGE_PANEL`              | 0.47    | spread = mid_PANEL_1X2 + 0.47 * mid_PANEL_2X2                       | S    |

Refit procedure (per day):
1. Slice in-sample mid-price series for the two legs.
2. Run OLS: `mid_A ~ beta * mid_B + alpha`.
3. Use `beta` (or `-beta`, depending on the spread sign convention) as the
   new hedge ratio constant.
4. Optionally, monitor stability across days. If the sign flips (as with
   `OXYGEN_SHAKES` last round), drop the pair.

---

## Recommended Search Budget

Categories are **independent** in PnL contribution (no cross-symbol coupling
in the strategies). Therefore sweep each category independently and **add**
their grid sizes; do not multiply across categories.

| Category    | Per-category grid size | Notes                                  |
|-------------|------------------------|----------------------------------------|
| Globals     | 3                      | Only `PASSIVE_MM_SIZE`.                |
| Pebbles     | 72                     | resid_trigger * passive_size * unwind. |
| Snack Packs | 192                    | window shared across pairs.            |
| Sleep Pods  | 240                    | window * 4 z-knobs.                    |
| Robots      | 240                    | window * 4 z-knobs.                    |
| UV Visors   | 240                    | window * 4 z-knobs.                    |
| Bronze      | 768                    | 4 pairs * 192 each (independent).      |
| Hedge ratios| 0 (refit, not gridded) | OLS each day.                          |

**Total backtest evaluations** (additive across categories): **3 + 72 + 192 + 240 + 240 + 240 + 768 ~= 1755 runs**.

If each backtest evaluation takes ~30 s of single-core compute, that is
~14.6 hours sequential, ~1.8 hours on 8 cores, ~30 min on 32 cores. Quite
tractable.

If budget is tighter, prioritise (in expected ROI order):
1. **Pebbles** (72 evals) — newly rewritten strategy with the most
   uncertainty; cheap to sweep.
2. **Sleep Pods + Snack Packs** (432 evals) — Tier 1/2 PnL drivers.
3. **Bronze windows only** (4^4 = 256, with z-knobs frozen) — cheap and the
   per-pair window is the main lever.
4. **Visors + Robots** if time allows.

Always re-fit hedge ratios first (Section 8) before any grid sweep — a
mis-specified spread invalidates the entire z-score calibration.
