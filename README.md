# Round 4 — Algorithmic Trading Strategy

## Status

`trader.py` is rebuilt on **Round 3's proven 5-layer market-making framework**,
with every parameter re-calibrated on Round 4 data.

**Backtest performance** (30 sliding windows of 1/10 day):

```
mean PnL/window: +13,305
Sharpe:           +0.59
hit rate:         25/30 = 83%
worst window:    -57,487
best window:     +49,229
```

vs. the earlier ML-driven approach at -340/window and a 30% hit rate.

## Strategy — Key Components

1. **5-layer passive market-making** — quotes posted around fair value at
   ±0.15σ / 0.30σ / 0.50σ / 0.80σ / 1.20σ, front-weighted 30/25/20/15/10%.
2. **Active mean-reversion take** — when the deviation from fair reaches ≥ 1σ,
   cross the spread on the favorable side.
3. **Session blending** — a session mean is accumulated from mids over the first
   2000 ticks; blending only engages once the deviation exceeds 0.3σ, so the
   static prior is not over-trusted when the backtest deviation is small.
4. **Position state machine** — `long_heavy` (>55% of limit) / `panic` (>70%)
   withdraw quotes on the over-weighted side and force inventory down.
5. **Order-book imbalance filter** — block adverse takes when top-of-book is
   imbalanced.
6. **Per-voucher handling**:
   - `VEV_4000` (deep ITM) → market-make off intrinsic value
   - `VEV_4500 / 5000 / 5100 / 5200` → direct mean-reversion to the historical mean
   - `VEV_5300 / 5400 / 5500` → signal from IV-smile residual
   - `VEV_6000 / 6500` (mid stuck at 0.5) → unwind only, no new positions
7. **Round 4 addition — counterparty nudge** — net buying by Mark 01 / 67 shifts
   fair up by 0.3σ; net selling by Mark 22 / 49 shifts it down.

## Key Parameters (top of `trader.py`)

```python
SMILE_COEF = (0.131776, 0.016920, 0.228592)  # from Round 4 EDA
VEV_FAIR = 5247.65;  VEV_STD = 16.73         # VELVETFRUIT_EXTRACT, 3-day intraday avg
HYDROGEL_FAIR = 9994.65;  HYDROGEL_STD = 34.06
ROUND_TTE_DAYS = 4
VEV_PROXY_PARAMS = { 4500:(747.66, 16.75), 5000:(251.14, 15.86), ... }
CP_NUDGE_FULL = 0.30  # max counterparty-signal fair shift (in σ units)
```

## Files

| File | Purpose |
|---|---|
| **`trader.py`** | **Competition submission** (Round 3 framework + Round 4 data) |
| `trader_50k_baseline.py` | Earlier 50k-baseline reference |
| `trader_47k_failed.py` | Failed variant kept for reference |
| `trader_round3 copy.py` | Round 3 original for reference (do not submit) |
| `backtest.py` | Local sliding-window backtest (1/10 day = 1 sim window) |
| `Round4-algo_trading_challenge_rules.md` | Round rules |
| `ROUND_4/*.csv` | Historical order-book and trades data |
| `eda_counterparty.py`, `eda_vouchers.py` | EDA scripts |
| `counterparty_findings.md`, `voucher_findings.md`, `cp_alpha_report.md` | EDA write-ups |
| `cp_features.py`, `train_cp_alpha.py`, `cp_alpha.json` | Counterparty-alpha feature/training pipeline |
| `counterparty_signals.csv`, `smile_snapshot.png` | EDA outputs |

## Local Validation

```bash
python3 backtest.py --days 1 2 3
```

Prints per-window PnL for 30 windows plus a sliding-window summary.

## Submission

Submit `trader.py` directly — it does not depend on any model file.

## Possible Next Steps

1. **Add a delta hedge** — hedge voucher directional risk with VELVETFRUIT.
2. **Tune `CP_NUDGE_FULL`** — invisible in backtest (the data set is roughly
   stationary) but may help live.
3. **Refit `SMILE_COEF`** — currently a 3-day average from Round 4 EDA; consider
   using only the most recent day.
4. **`VEV_PROXY_PARAMS` from the most recent day** — currently a 3-day average;
   daily trend may be more accurate.

---

# Round 4 — 算法交易策略

## 当前状态

`trader.py` 已基于 **Round 3 的成熟 5 层做市框架**重写，参数全部用 Round 4 数据重新校准。

**回测表现**（30 个 1/10 天 sliding window）：

```
mean PnL/window: +13,305
Sharpe:           +0.59
hit rate:         25/30 = 83%
worst window:    -57,487
best window:     +49,229
```

vs. 之前 ML 路线的 -340/win、hit 30%。

## 策略关键组件

1. **5 层被动做市**：在公允价 ±0.15σ ±0.3σ ±0.5σ ±0.8σ ±1.2σ 处分别挂单，front-weighted 30/25/20/15/10%
2. **均值回归主动吃单**：偏差 ≥ 1σ 时跨价吃 favorable side
3. **session blending**：前 2000 tick 累计 mid 计算 session mean，超过 0.3σ 偏差才 blend，避免 backtest 偏差小时过度 trust 静态 prior
4. **仓位状态机**：long_heavy（>55%）/ panic（>70%）→ 关闭对应方向挂单，强制减仓
5. **OB imbalance filter**：top-of-book 失衡时阻止逆向吃单
6. **Voucher 分类处理**：
   - `VEV_4000`（深 ITM）→ 用内在价值做市
   - `VEV_4500/5000/5100/5200` → 直接对历史 mean 做 mean-reversion
   - `VEV_5300/5400/5500` → 用 IV smile 残差做信号
   - `VEV_6000/6500`（mid 永远 0.5）→ 只 unwind，不开新仓
7. **Round 4 独家：counterparty nudge**：检测到 Mark 01/67 净买入 → 公允价上移 0.3σ；Mark 22/49 净卖出 → 下移

## 关键参数（在 `trader.py` 顶部）

```python
SMILE_COEF = (0.131776, 0.016920, 0.228592)  # 来自 Round 4 EDA
VEV_FAIR = 5247.65;  VEV_STD = 16.73        # VELVETFRUIT_EXTRACT 3 天 intraday 平均
HYDROGEL_FAIR = 9994.65;  HYDROGEL_STD = 34.06
ROUND_TTE_DAYS = 4
VEV_PROXY_PARAMS = { 4500:(747.66, 16.75), 5000:(251.14, 15.86), ... }
CP_NUDGE_FULL = 0.30  # CP 信号最大偏移倍率（用 σ 单位）
```

## 文件清单

| 文件 | 用途 |
|---|---|
| **`trader.py`** | **比赛提交文件**（基于 Round 3 框架 + Round 4 数据） |
| `trader_50k_baseline.py` | 早期 50k 基线参考 |
| `trader_47k_failed.py` | 失败变体，留作参考 |
| `trader_round3 copy.py` | Round 3 原版参考（请勿提交） |
| `backtest.py` | 本地滑窗回测（1/10 天 = 1 sim window） |
| `Round4-algo_trading_challenge_rules.md` | 规则文档 |
| `ROUND_4/*.csv` | 历史 orderbook 和 trades 数据 |
| `eda_counterparty.py`、`eda_vouchers.py` | EDA 脚本 |
| `counterparty_findings.md`、`voucher_findings.md`、`cp_alpha_report.md` | EDA 报告 |
| `cp_features.py`、`train_cp_alpha.py`、`cp_alpha.json` | 对手方 alpha 特征 / 训练流水线 |
| `counterparty_signals.csv`、`smile_snapshot.png` | EDA 产物 |

## 怎么本地验证

```bash
python3 backtest.py --days 1 2 3
```

输出 30 个 window 的 PnL 和滑窗 summary。

## 怎么提交

直接提交 `trader.py` 即可（不依赖任何模型文件）。

## 还可以试的方向

1. **加 delta hedge**：voucher 仓位用 VELVETFRUIT 对冲方向风险
2. **细调 CP_NUDGE_FULL**：backtest 看不出影响（数据集大致 stationary），live 可能有用
3. **重新拟合 SMILE_COEF**：当前用 Round 4 EDA 全 3 天平均，可考虑只用最近 1 天
4. **`VEV_PROXY_PARAMS` 用最近一天数据**：当前是 3 天平均，daily 趋势可能更准
