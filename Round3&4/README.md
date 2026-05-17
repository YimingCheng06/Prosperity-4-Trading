# Prosperity Round 3 & 4 — Algorithmic Trading

This branch holds the development work for **IMC Prosperity Rounds 3 and 4**.
Both rounds trade the **same instrument universe** — a delta-1 pair plus an
option chain — and share a common trading engine, so they are documented
together here.

## Instrument Universe

| Product | Limit | Type |
|---|---|---|
| `HYDROGEL_PACK` | 200 | delta-1 |
| `VELVETFRUIT_EXTRACT` (VFE) | 200 | delta-1 underlying |
| `VEV_4000` | 300 | deep-ITM call ≈ S − K |
| `VEV_4500 … 5500` | 300 each | option chain (vouchers) on VFE |
| `VEV_6000 / 6500` | 300 each | deep-OTM, no liquidity → unwind only |

Voucher time-to-expiry is **5 days** at the start of Round 3 and **4 days** for
Round 4.

## Shared Engine — Mean Reversion + 5-Layer Market Making

Every tradable product is reduced to a **fair value** and a **volatility scale
(σ)**, then handled by one routine:

1. **Microprice mid** — a size-weighted `(bid·ask_sz + ask·bid_sz)/(bid_sz+ask_sz)`
   instead of the naive mid; it leans away from the heavy side of the book to cut
   adverse selection.
2. **5-layer passive quotes** — bids/asks posted at ±0.15σ / 0.30σ / 0.50σ /
   0.80σ / 0.80σ with front-weighted size `[0.30, 0.25, 0.20, 0.15, 0.10]`. Inner
   layers actually sit at top-of-book and earn the spread; this front-weighting
   fixed a live "0 passive fills" bug where all PnL came from crossing the spread.
3. **Active mean-reversion take** — cross the spread when the deviation from fair
   reaches `TAKE_T = 1.0σ`; a wider reversal take fires at `REV_T = 2.25σ` with a
   position cap that scales with the deviation.
4. **Position state machine** — `heavy` at 55% of limit, `panic` at 70%:
   progressively withdraw quotes on the over-weighted side and force inventory
   back toward flat.
5. **Top-of-book imbalance filter** — block a take into a strongly adverse
   imbalance (`|I_top| > 0.5`), except when reducing a heavy/panic position.
6. **Anti-cross clipping** — bids clamped to ≤ best_ask−1, asks to ≥ best_bid+1,
   so inner layers never self-cross when fair drifts far from the mid.

Backtest-tuned constants: `TAKE_T = 1.0σ`, `REV_T = 2.25σ`, `heavy = 0.55·limit`,
`panic = 0.70·limit`, per-side quote size ≈ 50% of the limit.

## Round 3 — "Gloves Off"

**Approach.** Each VEV voucher (strikes 4500–5500) is treated as a *directly
mean-reverting series* against its own historically-calibrated mean — **not** as
an IV-smile arbitrage. Voucher mids turned out to have an even more stable daily
mean than the underlying itself, which became the round's single largest source
of alpha.

**Fair value.** Per-product mean/σ are calibrated offline (factor-mining
notebook) and then *session-blended* live: a partial-ramp blend of the static
prior with an online cumulative mean, gated so it only engages once live drift
exceeds 0.3σ, and frozen after 2000 ticks to avoid an EWMA trap (chasing
intraday drift kills mean-reversion alpha).

**Result.** 3-day merged backtest ≈ **747–762K**. Largest contributors:
HYDROGEL (~125K), VEV_5000 / 5100 / 5200 (~100K each), VFE (~79K). Deep-OTM
VEV_6000 / 6500 contribute ~0 and are unwound only.

**Confirmed dead ends** (do not re-add): EWMA / adaptive fair, portfolio delta
hedge, butterfly arbitrage on the voucher curve (−605K), single-layer quoting,
end-of-session de-risking, cross-strike position budgets.

## Round 4

Round 4 keeps the same products but adds three constraints and one new data set.

- **No hardcoded calibration** — per-product fair/σ may no longer be precomputed
  from history and embedded. Replaced by a **warmup-freeze**: estimate a trimmed
  mean over the first ~300 ticks, freeze it, then drift-correct with a ramped
  blend toward the online cumulative mean. `WARMUP_FREEZE_N = 300` must stay well
  below the 1000-tick live budget — otherwise the algorithm never trades.
- **Short-window σ correction** — a 300-tick warmup under-estimates σ, so it is
  inflated by `STD_INFLATE = 1.70`.
- **Named counterparties** — trades now expose buyer/seller identities
  ("Mark 01"–"Mark 67"). An event study profiled 7 actors: a large directional
  buyer (Mark 01) mirrored by a seller (Mark 22), several market-makers, and a
  one-sided buyer (Mark 67).

**Counterparty signal — what survived.** Single-actor follow-through is worth
only 1–2 ticks of edge while the spread is 6–10, so every cross-spread use of it
is −EV (an independent-take variant lost −883K). The only profitable use is a
**defensive take blocker**: skip an active take when that direction was just
initiated by an actor that historically front-runs it (+7.7K).

**ML pipeline.** A LightGBM + Ridge pipeline mined ~200 microstructure /
counterparty / cross-product features. Cumulative actor inventory (`cum_Mark_X`)
and the rolling time-premium (`tp_roll`) were the strongest predictors; Ridge
models reached out-of-sample Sharpe up to ~5. But a 200-tick forecast horizon
does not match a 5–50-tick MR engine — using the forecast as a *fair offset* or
a *take filter* both lost money (−195K, −170K). The only positive integration is
an **inventory-target bias**: the forecast sets a neutral position the MR engine
trades around (`RIDGE_TARGET_SCALE = 0.15`, +10K).

**Result.** The best Round 4 trader (`trader_round4_v4.py` — warmup-freeze +
counterparty blocker + Ridge inventory target) scores ≈ **472K** on the 3-day
backtest.

## Files

| File | Purpose |
|---|---|
| `trader_round3.py` | Round 3 submission |
| `trader_round4_v4.py` | Round 4 submission (warmup-freeze + counterparty blocker + Ridge target) |
| `trader_round4_D.py` | Round 4 fallback (no Ridge) |
| `trader_round4_base.py` | Round 4 minimal baseline (no Ridge, no counterparty) |
| `ROUND3_STRATEGY.md` | Round 3 full strategy & tuning log |
| `ROUND4_TECH_NOTES.md` | Round 4 technical reference (data stats, what worked / failed) |
| `Round_3_Gloves_Off.md` | Round 3 official rules |
| `round4_ridge_models.{py,json}` | Ridge model weights |
| `round4_alpha_rules.{py,csv}` | ML-distilled rule table |
| `colab_factor_mining.ipynb` | Round 3 calibration / factor mining |
| `colab_round4_counterparty.ipynb` | Round 4 ML pipeline |
| `ml_*.py`, `ml_*.csv` | Counterparty event-study scripts and outputs |
| `run_bt.py` | Backtest launcher (injects custom position limits) |
| `datamodel.py` | Platform data-model stub |

## Running backtests

```bash
# Round 3
python3 run_bt.py trader_round3.py 3 --no-progress --data bt_data --merge-pnl

# Round 4
python3 run_bt.py trader_round4_v4.py 4 --no-progress --data bt_data
```

> Backtest input data (`bt_data/`, `ROUND_3/`) is not version-controlled.

---

# Prosperity 第三轮 & 第四轮 — 算法交易

本分支保存 **IMC Prosperity 第三轮与第四轮**的开发工作。两轮交易**同一组商品** ——
一个 delta-1 配对加一条期权链 —— 并共用同一套交易引擎，因此合并在此说明。

## 商品清单

| 商品 | 持仓上限 | 类型 |
|---|---|---|
| `HYDROGEL_PACK` | 200 | delta-1 |
| `VELVETFRUIT_EXTRACT`（VFE） | 200 | delta-1 标的 |
| `VEV_4000` | 300 | 深度实值看涨 ≈ S − K |
| `VEV_4500 … 5500` | 各 300 | VFE 期权链（voucher） |
| `VEV_6000 / 6500` | 各 300 | 深度虚值，无流动性 → 仅平仓 |

Voucher 到期期限：第三轮开始时为 **5 天**，第四轮为 **4 天**。

## 共用引擎 —— 均值回归 + 5 层做市

每个可交易商品都被归结为一个**公允价**与一个**波动率尺度（σ）**，再由同一套
逻辑处理：

1. **Microprice 中间价** —— 用按挂单量加权的
   `(bid·ask_sz + ask·bid_sz)/(bid_sz+ask_sz)` 替代简单中间价，自动偏离订单簿
   较重的一侧，降低逆向选择。
2. **5 层被动报价** —— 在 ±0.15σ / 0.30σ / 0.50σ / 0.80σ / 0.80σ 处挂买/卖单，
   仓量前重 `[0.30, 0.25, 0.20, 0.15, 0.10]`。内层真正挂在盘口赚价差；这一前重
   权重修复了实盘"0 被动成交"的 bug（此前所有 PnL 都来自跨价吃单）。
3. **主动均值回归吃单** —— 当偏离公允价达到 `TAKE_T = 1.0σ` 时跨价吃单；更宽的
   反转吃单在 `REV_T = 2.25σ` 触发，仓位上限随偏差线性放大。
4. **仓位状态机** —— 仓位达上限 55% 为 `heavy`、70% 为 `panic`：逐步撤掉过重
   一侧的报价并强制把库存推回中性。
5. **盘口失衡过滤** —— 当出现强逆向失衡（`|I_top| > 0.5`）时屏蔽吃单，但在
   减仓（heavy/panic）时除外。
6. **防自成交裁剪** —— 买单裁剪至 ≤ best_ask−1、卖单 ≥ best_bid+1，避免公允价
   远离中间价时内层自相交。

回测调优常量：`TAKE_T = 1.0σ`、`REV_T = 2.25σ`、`heavy = 0.55·上限`、
`panic = 0.70·上限`，单边报价量约为持仓上限的 50%。

## 第三轮 —— "Gloves Off"

**思路。** 每个 VEV voucher（行权价 4500–5500）都被当作对自身历史均值的
*直接均值回归序列*，而**不是** IV 微笑套利。voucher 中间价的日内均值甚至比标的
本身更稳定，成为本轮最大的 alpha 来源。

**公允价。** 各商品的 mean/σ 离线校准（factor-mining notebook），实盘再做
*session 混合*：把静态先验与在线累计均值按 ramp 权重混合，并设门限——只有当
实盘漂移超过 0.3σ 时才启用——2000 tick 后冻结，避免 EWMA 陷阱（追逐日内漂移会
杀掉均值回归 alpha）。

**结果。** 三天合并回测约 **747–762K**。最大贡献：HYDROGEL（约 125K）、
VEV_5000 / 5100 / 5200（各约 100K）、VFE（约 79K）。深度虚值的 VEV_6000 / 6500
贡献约 0，仅做平仓。

**已验证无效**（不要再加回）：EWMA / 自适应公允价、组合 delta 对冲、voucher
曲线上的蝶式套利（−605K）、单层报价、临近收盘减仓、跨行权价仓位预算。

## 第四轮

第四轮沿用同一组商品，但新增了三项约束和一份新数据。

- **禁止硬编码校准** —— 不再允许把各商品的 fair/σ 从历史预算出来塞进代码。
  改为 **warmup-freeze**：用前约 300 tick 估一个截尾均值后冻结，再以 ramp 权重
  向在线累计均值做漂移修正。`WARMUP_FREEZE_N = 300` 必须远小于实盘 1000 tick
  预算，否则算法永远不会交易。
- **短窗 σ 修正** —— 300 tick 的 warmup 会低估 σ，故乘以 `STD_INFLATE = 1.70`。
- **具名对手方** —— 成交记录现在暴露买卖双方身份（"Mark 01"–"Mark 67"）。事件
  研究刻画了 7 个角色：一个大额方向性买家（Mark 01）与镜像卖家（Mark 22）、
  几个做市商，以及一个单边买家（Mark 67）。

**对手方信号 —— 留下来的部分。** 单一对手方的跟随效应只值 1–2 个 tick 的优势，
而价差是 6–10，因此任何跨价用法都是负 EV（独立吃单变体亏了 −883K）。唯一盈利
的用法是**防御性吃单屏蔽**：当某方向刚被历史上会抢跑它的对手方发起时，跳过
该方向的主动吃单（+7.7K）。

**ML pipeline。** 一条 LightGBM + Ridge 流水线挖掘了约 200 个微观结构 / 对手方 /
跨商品特征。对手方累计库存（`cum_Mark_X`）与滚动时间价值（`tp_roll`）是最强
预测因子，Ridge 模型的样本外 Sharpe 高达约 5。但 200-tick 的预测 horizon 与
5–50-tick 的均值回归引擎不匹配——把预测当作*公允价偏移*或*吃单过滤*都亏钱
（−195K、−170K）。唯一有正收益的整合方式是**库存目标偏置**：用预测设定一个
中性仓位，让均值回归引擎围绕它操作（`RIDGE_TARGET_SCALE = 0.15`，+10K）。

**结果。** 最佳的第四轮 trader（`trader_round4_v4.py` —— warmup-freeze +
对手方屏蔽 + Ridge 库存目标）在三天回测中约 **472K**。

## 文件说明

| 文件 | 用途 |
|---|---|
| `trader_round3.py` | 第三轮提交版 |
| `trader_round4_v4.py` | 第四轮提交版（warmup-freeze + 对手方屏蔽 + Ridge 目标） |
| `trader_round4_D.py` | 第四轮备用版（无 Ridge） |
| `trader_round4_base.py` | 第四轮最简基线（无 Ridge、无对手方） |
| `ROUND3_STRATEGY.md` | 第三轮完整策略与调参记录 |
| `ROUND4_TECH_NOTES.md` | 第四轮技术参考（数据统计、有效 / 无效清单） |
| `Round_3_Gloves_Off.md` | 第三轮官方规则 |
| `round4_ridge_models.{py,json}` | Ridge 模型权重 |
| `round4_alpha_rules.{py,csv}` | ML 蒸馏的规则表 |
| `colab_factor_mining.ipynb` | 第三轮校准 / 因子挖掘 |
| `colab_round4_counterparty.ipynb` | 第四轮 ML pipeline |
| `ml_*.py`、`ml_*.csv` | 对手方事件研究脚本与输出 |
| `run_bt.py` | 回测启动器（注入自定义持仓上限） |
| `datamodel.py` | 平台数据模型桩 |

## 跑回测

```bash
# 第三轮
python3 run_bt.py trader_round3.py 3 --no-progress --data bt_data --merge-pnl

# 第四轮
python3 run_bt.py trader_round4_v4.py 4 --no-progress --data bt_data
```

> 回测输入数据（`bt_data/`、`ROUND_3/`）未纳入版本控制。
