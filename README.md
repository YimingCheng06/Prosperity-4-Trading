# IMC Prosperity — Algorithmic Trading (Rounds 1–5)

This repository collects a full IMC Prosperity campaign — five trading rounds
across two planets. Each round's code, documentation, and rules are preserved in
its own folder; this README is the combined overview.

## Repository Layout

| Folder | Rounds | Contents |
|---|---|---|
| [`Round1&2/`](Round1%262) | 1 & 2 — Intara qualifiers | Reservation-price market-making and the first layered market-maker |
| [`Round3&4/`](Round3%264) | 3 & 4 — Solvenar options | Voucher mean-reversion, warmup-freeze, ML pipeline (development branch) |
| [`Round-4/`](Round-4) | 4 — alternate track | A separate, cleaned-up Round 4 implementation |
| [`Round5/`](Round5) | 5 — The Final Stretch | 50-product tiered per-category strategy |

Each folder keeps its own README with the full per-round detail. `Round3&4/` and
`Round-4/` both cover Round 4: the former is the development branch (Round 3
files plus several Round 4 trader iterations), the latter is a separate,
cleaned-up Round 4 implementation.

## The Common Thread — A Mean-Reversion + Layered Market-Making Engine

Every round prices each product as a **fair value** plus a **volatility scale
(σ)**, then trades it with the same family of mechanics, which grew round by
round:

- **Microprice mid** — a size-weighted mid that leans away from the heavy side of
  the book to reduce adverse selection.
- **Layered passive quotes** — multiple bid/ask layers around fair value,
  front-weighted so the inner layers earn the spread at top-of-book.
- **Mean-reversion active take** — cross the spread when price deviates far
  enough from fair value; a wider reversal take handles extreme dislocations.
- **Inventory state machine** — `heavy` / `panic` thresholds withdraw quotes on
  the over-weighted side and force inventory back toward flat.
- **Order-book imbalance filter** — block takes into a strongly adverse imbalance.

This engine first appeared as the 5-layer market-maker in **Round 2**, was tuned
and extended through Rounds 3–4 (session blending, warmup-freeze, counterparty
signals, Ridge ML), and informs the per-category strategies of Round 5.

## Round-by-Round

### Rounds 1 & 2 — Intara Qualifiers

Two products (`ASH_COATED_OSMIUM`, `INTARIAN_PEPPER_ROOT`, limit 80 each).
Round 1 prices ASH with a dynamic **reservation price** (mean-reversion pull +
order-book imbalance + microprice fade + inventory skew) and runs PEPPER as a
buy-and-hold trend long. Round 2 rebuilds ASH as a **5-layer market-maker** with
a `heavy`/`panic` inventory state machine and reversal bias — the seed of the
engine above. Result: **+8,012** (R1), **+7,682** (R2).

### Rounds 3 & 4 — Solvenar Options

A delta-1 pair (`HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`) plus a `VEV_*` option
chain. Round 3 treats each voucher as a **directly mean-reverting series** against
its own calibrated mean — voucher mids proved more stable than the underlying —
for a 3-day backtest of **~750K**. Round 4 keeps the products but bans hardcoded
calibration (replaced by a **warmup-freeze**), adds named-counterparty data (used
only as a defensive take blocker), and a LightGBM/Ridge ML pipeline (integrated
only as an inventory-target bias); best backtest **~472K**.

### Round 5 — The Final Stretch

50 brand-new products in 10 categories (limit 10 each). A **tiered per-category
strategy**: GOLD exploits a conservation law (Pebbles) and paired baskets (Snack
Packs); SILVER and BRONZE run cointegrated pair reversion; Oxygen Shakes are
skipped (non-stationary hedge ratios). Products without an edge fall back to a
universal market-making layer with microprice tilt.

---

# IMC Prosperity — 算法交易（第一至第五轮）

本仓库收录了一次完整的 IMC Prosperity 参赛过程 —— 跨越两颗星球的五个交易轮。
每一轮的代码、文档与规则都保存在各自的文件夹中；本 README 是综合总览。

## 仓库结构

| 文件夹 | 赛轮 | 内容 |
|---|---|---|
| [`Round1&2/`](Round1%262) | 第 1、2 轮 —— Intara 资格赛 | 保留价做市与首个分层做市引擎 |
| [`Round3&4/`](Round3%264) | 第 3、4 轮 —— Solvenar 期权 | Voucher 均值回归、warmup-freeze、ML pipeline（开发分支） |
| [`Round-4/`](Round-4) | 第 4 轮 —— 备选实现 | 一个独立、整理过的第四轮实现 |
| [`Round5/`](Round5) | 第 5 轮 —— The Final Stretch | 50 商品的分层分类策略 |

每个文件夹都保留了自己的 README，含该轮完整细节。`Round3&4/` 与 `Round-4/`
都涉及第四轮：前者是开发分支（第三轮文件加上数个第四轮 trader 迭代版本），
后者是一个独立、整理过的第四轮实现。

## 贯穿主线 —— 均值回归 + 分层做市引擎

每一轮都把每个商品归结为一个**公允价**加一个**波动率尺度（σ）**，再用同一族
机制交易；这套机制逐轮成长：

- **Microprice 中间价** —— 按挂单量加权的中间价，自动偏离订单簿较重的一侧，
  降低逆向选择。
- **分层被动报价** —— 在公允价附近挂多层买/卖单，仓量前重，使内层在盘口赚
  价差。
- **均值回归主动吃单** —— 当价格偏离公允价足够远时跨价吃单；更宽的反转吃单
  处理极端错价。
- **库存状态机** —— `heavy` / `panic` 阈值会撤掉过重一侧的报价，并强制把库存
  推回中性。
- **订单簿失衡过滤** —— 当出现强逆向失衡时屏蔽吃单。

这套引擎最早以 5 层做市的形态出现在**第二轮**，在第三、四轮被调优与扩展
（session 混合、warmup-freeze、对手方信号、Ridge ML），并影响了第五轮的
分类策略。

## 逐轮说明

### 第 1、2 轮 —— Intara 资格赛

两个商品（`ASH_COATED_OSMIUM`、`INTARIAN_PEPPER_ROOT`，各持仓上限 80）。
第一轮用动态**保留价**为 ASH 定价（均值回归拉力 + 订单簿失衡 + microprice
淡化 + 库存偏移），并把 PEPPER 当作买入持有的趋势做多。第二轮把 ASH 重建为
带 `heavy`/`panic` 库存状态机与反转偏置的 **5 层做市** —— 即上述引擎的雏形。
成绩：**+8,012**（第一轮）、**+7,682**（第二轮）。

### 第 3、4 轮 —— Solvenar 期权

一个 delta-1 配对（`HYDROGEL_PACK`、`VELVETFRUIT_EXTRACT`）加一条 `VEV_*`
期权链。第三轮把每个 voucher 当作对自身校准均值的**直接均值回归序列** ——
voucher 中间价比标的更稳定 —— 三天回测约 **750K**。第四轮沿用同一组商品，
但禁止硬编码校准（改用 **warmup-freeze**），引入具名对手方数据（仅用作防御性
吃单屏蔽），以及 LightGBM/Ridge ML pipeline（仅整合为库存目标偏置）；最佳
回测约 **472K**。

### 第 5 轮 —— The Final Stretch

10 个类别共 50 个全新商品（各持仓上限 10）。采用**分层分类策略**：GOLD 利用
守恒律（Pebbles）与配对篮子（Snack Packs）；SILVER 与 BRONZE 做协整配对
回归；Oxygen Shakes 跳过（hedge ratio 非平稳）。没有明显优势的商品由带
microprice 偏移的通用做市层兜底。
