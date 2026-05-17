# Round 3 "Gloves Off" — 完整策略与逻辑

**File**: `trader_round3.py`
**Backtest baseline**: 747,622（3-day merged）
**Last updated**: 2026-04-26

---

## 1. 产品与持仓上限

| Product | Limit | Type |
|---|---|---|
| HYDROGEL_PACK | 200 | underlying-like |
| VELVETFRUIT_EXTRACT (VFE) | 200 | underlying |
| VEV_4000 | 300 | deep ITM call ≈ S - K |
| VEV_4500 / 5000 / 5100 / 5200 / 5300 / 5400 / 5500 | 300 each | option chain on VFE |
| VEV_6000 / 6500 | 300 each | OTM tail，**unwind only** |

VEV strikes 4500-5500 都被建模成 **direct mean-reversion** （非 IV smile arb），因为 voucher mid 与 VFE mid level corr 0.99+，**daily mean 比 VFE 还稳定** —— 这是本 round 最大的 alpha 来源。

---

## 2. 核心校准常量

```python
# 来自 colab_factor_mining.ipynb 的 cleaned per-day demean fit
HYDROGEL_FAIR = 9989.4   ; HYDROGEL_STD = 31.92
VEV_FAIR      = 5255.4   ; VEV_STD      = 15.16   # 也用作 VEV_4000 = VEV_FAIR - K

VEV_PROXY_PARAMS = {  # (mean, std) 已 +15% 优化值
    4500: (750.0, 17.3),  5000: (255.0, 16.1),
    5100: (167.0, 13.8),  5200: ( 95.0, 10.4),
    5300: ( 46.5,  6.9),  5400: ( 16.0,  3.2),
    5500: (  6.6,  1.5),
}

# IV smile（仅用于深 ITM/OTM 的 unwind 价值估计；strikes 4500-5500 不走此路径）
SMILE_COEF      = (0.030889, 0.004210, 0.192393)   # IV(m) = a*m² + b*m + c
SMILE_RESID_STD = 0.007357
```

---

## 3. Session 校准机制（应对 live regime drift）

### 问题
live 1/10 day = 1000 ticks，每个 product 的 session_mean 系统性偏离 hardcoded fair **0.3-0.7σ**。
backtest 270 个 1000-tick window 的统计：77% drift ≥ 0.3σ，48% ≥ 0.6σ —— **drift 是结构性的**。

### 实现（`_blend_fair`）
```python
SESSION_FREEZE_N = 2000   # 实际 backtest 才到达；live 走 partial-ramp 路径
SESSION_RAMP_N   = 100    # tick=100 后达到 SESSION_MAX_W
SESSION_MAX_W    = 0.5    # mean blend 上限
SESSION_MAX_W_STD = 0.5   # std blend 上限（与 mean 解耦）
SESSION_GATE     = 0.3    # 偏差 < 0.3σ 时不 blend，避免噪声
```

每 tick 累积 `(sum, ssq, n)` of mid。
- n < 50：直接用 hardcoded fair
- n ∈ [50, FREEZE_N]：partial blend，`w_base = MAX_W * min(1, n/RAMP_N)`
- n ≥ FREEZE_N：冻结 sess_mean，永不更新（避免 EWMA 陷阱）

**Mean gate**：仅当 `|sess_mean - fair_static| ≥ GATE * std_static` 才启动 mean blend；否则 w_mean = 0。
**Std blend**：永远启用，`std_eff = max(0.3 * static, (1-w_std)*static + w_std*sess_std)`，floor 防止过窄。

### 为什么这样设计
- **不是 EWMA**：不持续追踪 intraday drift（会杀 alpha），只校准一次 regime level
- **gate**：backtest 偏差小（hardcoded 来自训练日），不 blend；live 偏差大，启动 blend
- **freeze**：避免 trader 自己的成交污染 sess_mean
- **stdlib json**（非 jsonpickle）：通过 `state.traderData` 持久化 session state

---

## 4. 通用 mean-reversion + 5 层做市（`_trade_mr`）

**所有 HYDROGEL / VFE / VEV_4000-5500 都走这条路径**。

### 4.1 Microprice mid
```python
mid = (best_bid * ask_size + best_ask * bid_size) / (bid_size + ask_size)
```
size-weighted 而非简单 mid，自动偏向 book 重的一侧，降低 adverse selection。

### 4.2 Top-of-book imbalance filter (I_top)
```python
I_top = (bv1 - av1) / (bv1 + av1)   # ∈ [-1, 1]
allow_buy_take  = (I_top >= -0.5) or short_heavy or panic_short
allow_sell_take = (I_top <=  0.5) or long_heavy  or panic_long
```
强逆向不平衡时屏蔽 take（防被 bot 反向选择）；heavy/panic 状态下跳过 filter（减仓优先）。

### 4.3 Std 缩放尺度
```python
L1 = round(std * 0.15)   # ~0.15σ 内层
L2 = round(std * 0.30)
L3 = round(std * 0.50)
L4 = round(std * 0.80)
L5 = round(std * 0.80)   # 与 L4 同距，留 outer ammo
TAKE_T = round(std * 1.00)   # 主动吃单阈值
REV_T  = round(std * 2.25)   # 反转跨价吃单阈值
```

### 4.4 Position 状态
```python
long_heavy   = position >  0.55 * limit    # 165 / 110
short_heavy  = position < -0.55 * limit
panic_long   = position >  0.70 * limit    # 210 / 140
panic_short  = position < -0.70 * limit
```

### 4.5 主动吃单
- 普通：买价 ≤ fair - TAKE_T；卖价 ≥ fair + TAKE_T
- long_heavy / panic_long：收紧买（fair - max(TAKE_T, L4)），放宽卖（fair - L1）
- short_heavy / panic_short：对称镜像

### 4.6 反转跨价吃单
当 |deviation| ≥ REV_T (=2.25σ) 且 position 方向允许时，跨价吃单。
```python
rev_cap = min(0.40 * limit, |dev|/std * limit * 0.30)
```
仓位 cap 随偏差线性增长，限制反向打满。

### 4.7 5 层挂单（passive market making）
```python
per_side = max(40, int(limit * 0.50))    # 总挂量 ~50% limit/side
layer_qtys = [int(per_side * w) for w in [0.3, 0.25, 0.2, 0.15, 0.1]]
# Front-weighted：inner layers 真正在 top of book 挂量赚 spread
# 旧 [0,0,0,0,1.0] 因 L5 通常 outside [bb+1, ba-1] clamp → 0 passive fills
# Live 462203 验证：旧版本 100% aggressive cross，spread leak -14.5K

bid_layers = zip([f-L1, f-L2, f-L3, f-L4, f-L5], layer_qtys)
ask_layers = zip([f+L1, f+L2, f+L3, f+L4, f+L5], layer_qtys)
```

### 4.8 反转偏置（quote skew）
- `deviation ≥ TAKE_T`：ask 收紧到 [f, f+L1, f+L2, f+L3, f+L4]，bid 外推
- `deviation ≤ -TAKE_T`：bid 收紧到 [f, f-L1, ...]，ask 外推

### 4.9 Heavy/Panic 仓位保护
```python
panic_long  → ask 集中 [(f, per_side/2), (f+L1, per_side/2)]，bid 全删
long_heavy  → 5 层 ask 各 per_side/4，bid 仅 (f-L4, per_side/4)
panic_short → 镜像
short_heavy → 镜像
```

### 4.10 防 cross 边界 clipping（关键 bug fix）
```python
max_bid = best_ask - 1
min_ask = best_bid + 1
# 提交 bid 前必须 ≤ max_bid；ask 必须 ≥ min_ask
```
旧版本 inner layer 在 `fair << mid` 时会 cross-sell at bid，造成大量 adverse fill。

---

## 5. 顶层 dispatch（`run`）

```python
for product in state.order_depths:
    if product == "HYDROGEL_PACK":
        fair, std = blend_fair(HYDROGEL_FAIR, HYDROGEL_STD)
        orders = trade_mr(...)
    elif product == "VELVETFRUIT_EXTRACT":
        fair, std = blend_fair(VEV_FAIR, VEV_STD)
        orders = trade_mr(...)
    elif product == "VEV_4000":
        # 深 ITM ≈ S - K，复用 VEV_STD
        fair, std = blend_fair(VEV_FAIR - 4000, VEV_STD)
        orders = trade_mr(...)
    elif K in {4500..5500}:
        fair_K, std_K = VEV_PROXY_PARAMS[K]
        fair, std = blend_fair(fair_K, std_K)
        orders = trade_mr(...)
    elif K in {6000, 6500}:
        orders = trade_voucher(...)   # OTM tail：unwind only
```

VEV_6000/6500 走 `_trade_voucher`（IV smile residual），但实际只 unwind position（不主动建仓），因 OTM 流动性差且 PnL 贡献近零。

---

## 6. 已验证的关键 alpha 点（按贡献排序）

1. **VEV vouchers 直接 MR**（+226K vs 旧 IV smile arb）
   voucher mid 比 VFE mid 的 daily mean 还稳定，对各自 hardcoded mean 直接做 MR 是不同范式。

2. **TAKE_T = 1.0σ**（+24K vs 0.6σ）
   原 0.6σ 太激进吃噪声；1.0σ 是局部最优。

3. **Heavy=0.55 / Panic=0.70**（+32K vs 0.30/0.55）
   旧 heavy=0.30 提前缩 take 区间杀 alpha；放宽后 position 跑得满。

4. **Microprice mid**（+18K）
   size-weighted mid 自动偏向重的一侧。

5. **VEV_PROXY_PARAMS std +15%**（+5K）
   day 2 V-shape 实际波动比训练日大。

6. **REV_T = 2.25σ**（+16K vs 1.0σ）
   ≥ 2.0σ 后实际很少触发，但作为 safety net 保留。

7. **Front-weighted layer_qtys**（live: -14.5K spread leak → 正 spread；BT: -3K）
   修复 0 passive fills 的 live bug。

8. **GATE=0.3, RAMP_N=100**（live: blend engage ; BT: -12K）
   live drift 0.3-0.7σ 全在 gate 范围内。

---

## 7. 已尝试且确认无效（不要再加回）

| 改动 | 结果 | 原因 |
|---|---|---|
| EWMA / cumulative mean | 杀 alpha | 跟踪 intraday drift 等于跟踪噪声 |
| Portfolio delta hedge | 赔钱 | vouchers 自己 MR，强行偏置反而扭曲 fair |
| 内重 layer [1,0,0,0,0] | -92K | 集中一层无法吸收波动 |
| Butterfly arb on B(K) | -605K disaster | 与 per-voucher MR 抢同一 alpha + 跨价太贵 |
| Adaptive fair α=0.0005/0.00005 | 退步 | EMA 追 intraday drift，落入 EWMA 陷阱 |
| TAKE_T 0.4σ / 0.6σ / 1.2σ | 都退步 | 1.0σ 是窄区间最优 |
| Voucher mm_qty 30→60 | 无效 | mm_bid 被 best_bid 限制是 dead code |
| HYDROGEL-VEV pair trade | 无效 | tick-diff corr ≈ 0 |
| 5300-5500 theta-aware fair | -17K | hardcoded mean 已包含 theta，加 ramp 退步 |
| VEV_STD → 18 | 退步 | day 1 -5K vs day 2 +3K，整体负 |
| inv_skew k=0.2σ + warmup 1500 | -30K | 实盘走势没改善 |
| Chain-drift (VFE → VEV propagation) | -210K BT | drift 信号在 BT 是噪声 |
| Cross-VEV shared budget (cap 600-2100) | -16K~-370K | 限制了正常 hedging |
| End-of-session de-risk (linear ramp) | -100K+ | 早期就开始减仓杀 alpha |
| DD circuit breaker | 无用 | 阈值高不触发，阈值低杀 alpha |
| Smaller VEV limit (250/200) | -83K~-170K | 直接砍 capacity |

---

## 8. 当前 per-product PnL 贡献（BT 762K baseline）

| Product | PnL | 角色 |
|---|---|---|
| HYDROGEL_PACK | 125K | 最大单产品（drawdown -19K） |
| VEV_5000 | 106K | 核心 voucher MR alpha |
| VEV_5100 | 104K | 核心 voucher MR alpha |
| VEV_5200 | 86K | 核心 voucher MR alpha |
| VELVETFRUIT_EXTRACT | 79K | underlying MM |
| VEV_4500 | 72K | proxy MR |
| VEV_5300 | 56K | proxy MR |
| VEV_4000 | 55K | deep ITM ≈ S-K |
| VEV_5400 | 22K | 边缘 |
| VEV_5500 | 3.7K | 边缘 |
| VEV_6000 / 6500 | 0 | unwind only |

---

## 9. Live vs Backtest 巨差解释

**Live 462203（1/10 day 提交）**: PnL +26,605
**Backtest 同等量级（×10）**: 762K → 应 ~76K

**主因（Agent D 重新诊断）**：
- 100% 的 fill 都是 aggressive cross（348/359）
- 0 passive fills
- spread cost -14,532 ≈ 整个 give-back（peak 31K → end 26K）

**根因**：旧 `layer_qtys=[0,0,0,0,1.0]` 让 inner layers (L1-L4) qty=0，仅 L5 posting；但 L5 通常落在 `[best_bid+1, best_ask-1]` clamp 外被 filter → **完全不挂 passive**。所有 PnL 来自 take/REV crossing spread。

**已 fix**：layer_qtys 改前重 [0.3, 0.25, 0.2, 0.15, 0.1]。Backtest 影响微小（-3K），live 预期把 -14.5K spread leak 转为正 spread 收入。

**次因**：session-mean drift 0.3-0.7σ 时旧 GATE=0.6 不启动 blend，导致 fair 永远 stale。已 fix（GATE=0.3）。

**Backtest 不模拟 bot reaction**：prosperity3bt 不会让 bot 主动反向跑，所以 backtest 的 layer/take 比例与 live 完全不同。这是 28K vs 76K 残余 gap 的主要解释。

---

## 10. 文件结构

```
/Users/emmett/Documents/Prosperity Trading/
├── trader_round3.py              # 提交版
├── ROUND3_STRATEGY.md            # 本文档
├── colab_factor_mining.ipynb     # ML alpha mining + cleaned 校准
├── run_bt.py                     # backtest launcher（注入 LIMITS）
├── pnl_breakdown.py              # per-product PnL 工具
├── ROUND_3/                      # 训练数据（CSV）
│   ├── prices_round_3_day_{0,1,2}.csv
│   └── trades_round_3_day_{0,1,2}.csv
├── bt_data/round3/               # backtest 输入
└── 462203.log                    # 最近 live 提交日志
```

---

## 11. 跑 backtest

```bash
cd "/Users/emmett/Documents/Prosperity Trading"
python3 run_bt.py trader_round3.py 3 --no-progress --data bt_data --merge-pnl
```

`run_bt.py` 在调用 `prosperity3bt` 前注入 HYDROGEL_PACK / VELVETFRUIT_EXTRACT / VEV_* 到 `LIMITS` dict（这些产品不在原 prosperity3bt 默认列表里）。

---

## 12. 调参原则（防再次过拟合）

1. **超参数变更必须 backtest 验证**：sweep 一个区间，看是否单调 / 有平台
2. **Live 数据只用作"是否引入未来函数"的 sanity check**，不能直接 fit hyperparam（除非有多个 live sample）
3. **结构性 fix（如 layer_qtys 前重）优先于参数调优**：前者有数据机制支撑，后者易过拟合
4. **memory 中的"已尝试无效"列表必读**：避免重复同样死路
