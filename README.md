# Prosperity Round 5 — Algorithmic Trading

IMC Prosperity *"The Final Stretch"* — the fifth and final round. It introduces
**50 brand-new products**, evenly split into **10 categories of 5**, each with a
hard position limit of **10**. The challenge theme, **"Cherry Picking Winners"**,
is that some categories hide exploitable structural patterns while the rest only
support generic market-making.

## Strategy Overview

All trading logic lives in a single submission file, `trader_baseline.py`,
organised as a **tiered per-category strategy**:

| Tier | Category | Edge exploited |
|---|---|---|
| GOLD | Purification Pebbles | Linear conservation `XS+S+M+L+XL = 50000`; residual-conditioned passive MM on single legs |
| GOLD | Protein Snack Packs | Two paired baskets (CHOC+VAN, STRAW+RASP), z-score mean reversion |
| SILVER | Vertical Sleeping Pods | Cointegrated pairs POLY ≈ 0.964·COTTON, LAMB ≈ 0.401·NYLON |
| SILVER | Domestic Robots | DISHES vs an equal-weight basket of the other four robots |
| SILVER | UV-Visors | AMBER / MAGENTA pair reversion |
| BRONZE | Galaxy Sounds, Microchips, Translators, Panels | Small-size pair reversion, capped at \|pos\| ≤ 3 |
| NOPE | Liquid Breath Oxygen Shakes | Skipped — hedge ratios flip sign across days, no stable structure |

### Design rationale

- **Pebbles** — the five sizes sum to a constant 50000 *even at the quote level*
  (`sum_ask ≈ sum_bid ≈ 50000`), so a take-arbitrage on the whole basket is
  impossible and joint passive fills have ~0 probability. The strategy instead
  runs residual-conditioned single-leg passive market-making.
- **Pair / basket trades** — each pair tracks a spread `leg_A + β·leg_B`;
  positions open on a z-score threshold and unwind toward the mean, sized
  linearly in the deviation.
- **Universal MM fallback** — products not covered by a tier strategy are quoted
  by a passive MM layer that skews quotes by order-book imbalance (microprice tilt).
- **Oxygen Shakes** are deliberately left flat: the hedge ratio is non-stationary
  across the three data days, so any pair model would be fitting noise.

## Files

| File | Purpose |
|---|---|
| `trader_baseline.py` | Single-file submission for the Prosperity platform |
| `TUNABLE_PARAMS.md` | Every tunable hyperparameter, grid ranges, hedge-ratio notes |
| `backtest_round5.ipynb` | Colab backtester — parameter grid search, OLS hedge-ratio refit |
| `backtest_round5_colab.pdf` | Exported Colab run of the backtest notebook |
| `smoke_test.py` | Drives real data through `Trader.run`; checks for exceptions and position-limit violations |
| `round5_rules.md` | Round rules and the 50-product list |

## Usage

Smoke test (needs the `ROUND_5/` data directory and a `prosperity3bt` environment):

```bash
python smoke_test.py
```

Backtesting and parameter grid search: open `backtest_round5.ipynb` in Colab.

> Competition CSVs (`ROUND_5/`, ~110 MB) are not version-controlled — fetch them separately.

## API

```python
Trader().run(state) -> (orders_dict, conversions_int, traderData_str)
```

---

# Prosperity 第五轮 — 算法交易

IMC Prosperity *"The Final Stretch"* 第五轮，也是最后一轮。本轮引入
**50 个全新商品**，均匀分为 **10 个类别（每类 5 个）**，每个商品持仓上限均为
**10**。挑战主题为 **"Cherry Picking Winners"**：部分类别藏有可被利用的结构性
模式，其余类别则只能依靠通用做市策略。

## 策略概览

全部交易逻辑集中在单文件 `trader_baseline.py`，按类别分层：

| 层级 | 类别 | 利用的优势 |
|---|---|---|
| GOLD | Purification Pebbles | 线性守恒 `XS+S+M+L+XL = 50000`；残差条件下的单腿被动做市 |
| GOLD | Protein Snack Packs | 两组配对篮子（CHOC+VAN、STRAW+RASP）做 z-score 均值回归 |
| SILVER | Vertical Sleeping Pods | 协整配对 POLY ≈ 0.964·COTTON、LAMB ≈ 0.401·NYLON |
| SILVER | Domestic Robots | DISHES 对其余 4 个机器人的等权篮子 |
| SILVER | UV-Visors | AMBER / MAGENTA 配对回归 |
| BRONZE | Galaxy Sounds、Microchips、Translators、Panels | 小仓位配对回归，`\|pos\| ≤ 3` |
| NOPE | Liquid Breath Oxygen Shakes | 跳过 —— hedge ratio 跨日变号，无稳定结构 |

### 设计依据

- **Pebbles** —— 五个尺寸之和恒为 50000，且**在报价层面也成立**
  （`sum_ask ≈ sum_bid ≈ 50000`），因此整组 take 套利不可行、joint passive
  fill 概率近 0。策略改为残差条件下的单腿被动做市。
- **配对 / 篮子交易** —— 每个配对跟踪价差 `leg_A + β·leg_B`；当 z-score
  越过阈值时开仓，向均值回归时平仓，仓位按偏差线性缩放。
- **通用做市兜底** —— 未被分层策略覆盖的商品由被动做市层报价，并按订单簿
  失衡做报价偏移（microprice tilt）。
- **Oxygen Shakes** 刻意保持空仓：其 hedge ratio 在三天数据间非平稳，任何配对
  模型都是在拟合噪声。

## 文件说明

| 文件 | 用途 |
|---|---|
| `trader_baseline.py` | 提交到 Prosperity 平台的单文件交易程序 |
| `TUNABLE_PARAMS.md` | 全部可调超参清单、网格范围与 hedge ratio 标注 |
| `backtest_round5.ipynb` | Colab 回测 notebook —— 参数网格搜索、OLS hedge ratio 拟合 |
| `backtest_round5_colab.pdf` | 回测 notebook 的 Colab 运行结果导出 |
| `smoke_test.py` | 用真实数据驱动 `Trader.run`，校验无异常、无持仓违例 |
| `round5_rules.md` | 本轮规则与 50 个商品清单 |

## 使用方法

烟雾测试（需要 `ROUND_5/` 数据目录与 `prosperity3bt` 环境）：

```bash
python smoke_test.py
```

回测与参数网格搜索：在 Colab 中打开 `backtest_round5.ipynb`。

> 比赛数据 CSV（`ROUND_5/`，约 110 MB）未纳入版本控制，需另行获取。

## API

```python
Trader().run(state) -> (orders_dict, conversions_int, traderData_str)
```
