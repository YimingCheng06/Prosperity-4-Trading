# Prosperity Round 1 & 2 — Algorithmic Trading

This branch holds the development work for **IMC Prosperity Rounds 1 and 2** —
the two qualifier rounds set on the planet **Intara**. Both rounds trade the
**same two products**, so they are documented together here.

## Round Context

- **Round 1 — "Trading groundwork"** opens the mission: establish a Trade
  Outpost and reach a net profit of 200,000 XIRECs before the third trading day.
- **Round 2 — "Growing Your Outpost"** is the second and final qualifier. It
  keeps the same products and adds an optional **Market Access Fee** auction — a
  `bid()` function that competes for 25% extra order-book quotes (top 50% of bids
  win). This trader does not implement `bid()`, i.e. it runs on standard market
  access.

## Instrument Universe

| Product | Limit | Behaviour |
|---|---|---|
| `ASH_COATED_OSMIUM` | 80 | Volatile, mean-reverting around ~10000 |
| `INTARIAN_PEPPER_ROOT` | 80 | Steady, slow upward drift |

## Round 1 — `trader_round1.py`

### ASH_COATED_OSMIUM — reservation-price market-making

The fair value is a **reservation price** that blends four adjustments around a
long-run anchor of 10000:

```
reservation = 10000
            + 0.15 · (10000 − mid)        # mean-reversion pull to the anchor
            + 3.0  · imbalance            # lean toward the heavier side of L1
            − 0.8  · (microprice − mid)   # fade short-term microprice deviation
            − 0.05 · position             # inventory skew
```

where `imbalance = (bid_vol − ask_vol) / (bid_vol + ask_vol)` and `microprice` is
the L1-volume-weighted price.

- **Active take** — buy the best ask when it is ≤ `reservation − 0.5`; sell the
  best bid when it is ≥ `reservation + 0.75` (asymmetric thresholds, top level
  only, to avoid over-aggressive fills).
- **Passive quotes** — two layers per side at `reservation ± 1` and `± 2` with
  sizes 15 / 25, clamped inside the spread. Posting on a side stops once the
  position passes ±55, so the book is never pushed further into a heavy position.

### INTARIAN_PEPPER_ROOT — trend-following long

A steady, slowly-appreciating product, so the strategy is deliberately simple:
**buy up to the full 80-unit limit as fast as possible and hold**, capturing the
drift. It never market-makes and never sells, so trend profit is not given back.

## Round 2 — `trader_round2.py`

Round 2 refines the ASH strategy into the **5-layer market-maker with an
inventory state machine** that later rounds build on. `INTARIAN_PEPPER_ROOT` is
unchanged.

### ASH_COATED_OSMIUM — 5-layer MM + reversal bias + inventory state machine

Anchored at `FAIR_CENTER = 10000` (the 3-day statistical mean, σ ≈ 5):

- **Inventory state machine** — `heavy` at ±20 of the limit, `panic` at ±60.
- **Inventory-aware taking** — the take window shifts with the state; e.g. when
  `panic_long`, it will even sell below fair value to shed inventory fast.
- **Reversion take** — when `|mid − fair| ≥ 4`, actively cross the spread to
  build a reversal position, capped by `rev_cap = min(70, 20 + 6·|deviation|)`
  so the cap scales with the strength of the dislocation.
- **5-layer passive quotes** — base layers at `fair ± (1, 2, 3, 5, 7)` with sizes
  `(10, 20, 20, 15, 15)`.
- **Reversal bias** — when `|deviation| ≥ 3`, quotes on the reversion-favorable
  side are tightened inward while the opposite side is pushed outward.
- **Position protection** — in a `heavy`/`panic` state the layers collapse to a
  one-sided, inventory-reducing quote stack.

## Strategy Evolution — Round 1 → Round 2

| Aspect | Round 1 | Round 2 |
|---|---|---|
| ASH fair value | Dynamic reservation price | Fixed anchor (3-day mean) + deviation logic |
| ASH passive quotes | 2 layers per side | 5 layers per side |
| Inventory control | Single ±55 posting cutoff | `heavy` / `panic` state machine |
| Reversal handling | None (take only) | Reversal bias + reversal take |
| PEPPER | Trend-following long | Unchanged |

Round 2's layered market-maker with a `heavy`/`panic` state machine became the
core engine reused and extended in the later rounds of this project.

## Results

| Round | Submission | PnL |
|---|---|---|
| Round 1 | `189866` | +8,012.81 XIRECs |
| Round 2 | `333599` | +7,682.13 XIRECs |

## Files

| File | Purpose |
|---|---|
| `trader_round1.py` | Round 1 submission (reservation-price MM + trend long) |
| `trader_round2.py` | Round 2 submission (5-layer MM + state machine + trend long) |
| `Round1-Trading_Groundwork.pdf` | Round 1 official rules |
| `Round2-Growing_Your_Outpost.pdf` | Round 2 official rules |

---

# Prosperity 第一轮 & 第二轮 — 算法交易

本分支保存 **IMC Prosperity 第一轮与第二轮**的开发工作 —— 这两轮是发生在
**Intara** 星球上的资格赛。两轮交易**同一组两个商品**，因此合并在此说明。

## 赛轮背景

- **第一轮 ——"Trading groundwork"** 开启任务：建立交易前哨站，并在第三个
  交易日之前达到 200,000 XIRECs 的净利润。
- **第二轮 ——"Growing Your Outpost"** 是第二轮、也是最后一轮资格赛。商品
  不变，新增了可选的 **Market Access Fee** 拍卖 —— 一个 `bid()` 函数，用于
  竞争 25% 的额外订单簿报价（出价前 50% 中标）。本 trader 未实现 `bid()`，
  即按标准市场准入运行。

## 商品清单

| 商品 | 持仓上限 | 特性 |
|---|---|---|
| `ASH_COATED_OSMIUM` | 80 | 波动较大，围绕 ~10000 均值回归 |
| `INTARIAN_PEPPER_ROOT` | 80 | 平稳，缓慢上行漂移 |

## 第一轮 —— `trader_round1.py`

### ASH_COATED_OSMIUM —— 保留价做市

公允价是一个**保留价（reservation price）**，在长期锚点 10000 附近融合四项
调整：

```
reservation = 10000
            + 0.15 · (10000 − mid)        # 向锚点的均值回归拉力
            + 3.0  · imbalance            # 偏向 L1 较重的一侧
            − 0.8  · (microprice − mid)   # 淡化短期 microprice 偏离
            − 0.05 · position             # 库存偏移
```

其中 `imbalance = (bid_vol − ask_vol) / (bid_vol + ask_vol)`，`microprice` 为
按 L1 挂单量加权的价格。

- **主动吃单** —— 当最优卖价 ≤ `reservation − 0.5` 时买入；当最优买价
  ≥ `reservation + 0.75` 时卖出（阈值非对称，仅吃最优一档，避免过度激进）。
- **被动报价** —— 每边两层，价格在 `reservation ± 1` 与 `± 2`，仓量 15 / 25，
  并裁剪在价差之内。当仓位越过 ±55 后停止该方向报价，避免把仓位推得更重。

### INTARIAN_PEPPER_ROOT —— 趋势做多

这是一个平稳、缓慢升值的商品，因此策略刻意从简：**尽快买满 80 单位的上限并
持有**，吃掉趋势利润。它不做市、也不卖出，因此趋势利润不会回吐。

## 第二轮 —— `trader_round2.py`

第二轮把 ASH 策略精炼成后续各轮都沿用的 **5 层做市 + 库存状态机**。
`INTARIAN_PEPPER_ROOT` 保持不变。

### ASH_COATED_OSMIUM —— 5 层做市 + 反转偏置 + 库存状态机

锚定在 `FAIR_CENTER = 10000`（三天统计均值，σ ≈ 5）：

- **库存状态机** —— 仓位达上限 ±20 为 `heavy`、±60 为 `panic`。
- **库存感知吃单** —— 吃单窗口随状态平移；例如处于 `panic_long` 时，甚至会
  在公允价以下卖出以快速减仓。
- **反转吃单** —— 当 `|mid − fair| ≥ 4` 时主动跨价建立反转仓位，上限为
  `rev_cap = min(70, 20 + 6·|偏差|)`，使上限随错价强度放大。
- **5 层被动报价** —— 基础层在 `fair ± (1, 2, 3, 5, 7)`，仓量
  `(10, 20, 20, 15, 15)`。
- **反转偏置** —— 当 `|偏差| ≥ 3` 时，对反转有利一侧的报价向内收紧，另一侧
  向外推。
- **仓位保护** —— 进入 `heavy`/`panic` 状态时，报价层收缩为单边的减仓报价。

## 策略演进 —— 第一轮 → 第二轮

| 方面 | 第一轮 | 第二轮 |
|---|---|---|
| ASH 公允价 | 动态保留价 | 固定锚点（三天均值）+ 偏差逻辑 |
| ASH 被动报价 | 每边 2 层 | 每边 5 层 |
| 库存控制 | 单一 ±55 报价截止 | `heavy` / `panic` 状态机 |
| 反转处理 | 无（仅吃单） | 反转偏置 + 反转吃单 |
| PEPPER | 趋势做多 | 不变 |

第二轮带有 `heavy`/`panic` 状态机的分层做市引擎，成为本项目后续各轮复用并
扩展的核心引擎。

## 成绩

| 赛轮 | 提交编号 | PnL |
|---|---|---|
| 第一轮 | `189866` | +8,012.81 XIRECs |
| 第二轮 | `333599` | +7,682.13 XIRECs |

## 文件说明

| 文件 | 用途 |
|---|---|
| `trader_round1.py` | 第一轮提交版（保留价做市 + 趋势做多） |
| `trader_round2.py` | 第二轮提交版（5 层做市 + 状态机 + 趋势做多） |
| `Round1-Trading_Groundwork.pdf` | 第一轮官方规则 |
| `Round2-Growing_Your_Outpost.pdf` | 第二轮官方规则 |
