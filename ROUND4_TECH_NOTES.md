# Round 4 — Technical Reference for Rewrite

> 综合本轮全部调研、ML pipeline、backtest sweep 后的事实清单。
> 用于从零重写 trader 时直接查阅，避免重复踩坑。
> Last updated: 2026-04-27

---

## 0. 比赛规则约束（硬性）

| 项 | 值/说明 |
|---|---|
| 产品 | `HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`, `VEV_{4000,4500,5000,5100,5200,5300,5400,5500,6000,6500}` |
| 持仓上限 | HYDROGEL=200, VFE=200, 每个 VEV=300 |
| TTE | Voucher 在 R4 中 TTE = 4 天（与 R3 不同） |
| 新增字段 | `Trade.buyer` / `Trade.seller` 现在是真名（"Mark 01"–"Mark 67"） |
| **No hardcode** | 用户硬性要求：不能把 per-product fair / std 等从历史数据预算出来塞进代码 |
| **No future function** | 任何特征不能用 `t > now` 的数据 |
| **No `import os`** | 提交沙箱 forbidden patterns 包括 `import\s*os`，会被拒（regex） |
| **Live tick 预算** | **实盘提交 = 1000 ticks（1/10 day）**。Backtest = 10000 ticks/day × 3 days |

> ⚠️ 任何 `WARMUP_FREEZE_N ≥ 1000` → 实盘永远不交易 → PnL=0。R3/R4 都犯过这个 bug。

---

## 1. R4 数据基本统计（3 days × 10000 ticks）

### Per-day mean / std

| 产品 | d1 mean | d2 mean | d3 mean | d-d 漂移 (σ) | d1 std | d3 std |
|---|---|---|---|---|---|---|
| HYDROGEL_PACK | 9991.46 | 9990.79 | **10002.91** | 0.42 | 32.0 | 28.5 |
| VELVETFRUIT_EXTRACT | 5248.97 | 5255.80 | 5238.99 | 1.25 | 11.6 | 14.5 |
| VEV_4000 | 1248.98 | 1255.82 | 1238.93 | 1.26 | 11.6 | 14.5 |
| VEV_4500 | 748.88 | 755.83 | 738.89 | 1.27 | 11.6 | 14.4 |
| VEV_5000 | 253.72 | 259.04 | 241.35 | 1.40 | 10.7 | 13.8 |
| VEV_5100 | 165.47 | 167.65 | 149.86 | 1.58 | 9.4 | 12.0 |
| VEV_5200 | 95.48 | 94.53 | **77.37** | 2.13 | 6.9 | 8.8 |
| VEV_5300 | 47.13 | 44.71 | **31.76** | 3.00 | 4.4 | 4.9 |
| VEV_5400 | 15.67 | 13.70 | **8.14** | 3.22 | 2.3 | 2.1 |
| VEV_5500 | 6.58 | 5.35 | **2.15** | 4.34 | 1.1 | 0.7 |
| VEV_6000 | 0.50 | 0.50 | 0.50 | — | **0** | **0** |
| VEV_6500 | 0.50 | 0.50 | 0.50 | — | **0** | **0** |

**结论**：
- HYDROGEL 跨日漂移最小（0.42σ）
- VEV_5200+ 跨日漂移大（>2σ），主要是 **theta decay**（TTE=4d 越接近到期价值越低）
- VEV_6000/6500 完全没流动性 → **直接 skip 或 unwind only**

### Voucher-VFE 关系（关键洞察）

```
time_premium(K) := voucher_mid + K - VFE_mid (= max(0, S-K) 的时间价值部分)
```

| K | d1 tp_mean | d3 tp_mean | tp std | 跨日 tp 漂移 (σ) |
|---|---|---|---|---|
| 4000 | **0.02** | **0.00** | 0.84-0.91 | **0.02 ≈ 0** |
| 4500 | **0.01** | **0.00** | 0.76-0.79 | **0.02 ≈ 0** |
| 5000 | 4.87 | 2.46 | 1.0-1.3 | 2.14 |
| 5100 | 16.59 | 11.12 | 2.4-3.3 | 1.88 |
| 5200 | 46.74 | 38.64 | 5.9-7.2 | 1.25 |
| 5300 | 98.52 | 92.98 | 9.3-12.3 | 0.89 |

**应用**：
- VEV_4000 / VEV_4500 时间价值 ≈ 0 → fair = `max(VFE_mid - K, 0)` 直接成立，不需要单独 warmup（但是这个法不能直接用作 MR anchor —— 见 §6.A 失败实验）
- VEV_5000+ 需要在线估计 time_premium，跨日漂移大

### Voucher-VFE diff correlation

每天 voucher_mid_diff 与 VFE_mid_diff 的相关性 / β：

| K | corr | β |
|---|---|---|
| 4000 | 0.59 | 0.74 |
| 4500 | 0.60 | 0.67 |
| 5000 | 0.75 | 0.66 |
| 5100 | 0.77 | 0.59 |
| 5200 | 0.73 | 0.43 |
| 5300 | 0.62 | 0.27 |

---

## 2. Counterparty 画像（4281 trades, 7 actors）

### 角色分类

| Actor | 总买 | 总卖 | 净 | 角色 |
|---|---|---|---|---|
| **Mark 01** | 6053 | 1375 | **+4678** | 大户买家（VFE + 高 strike voucher） |
| **Mark 22** | 206 | 5683 | **-5477** | 大户卖家（与 Mark 01 镜像） |
| Mark 14 | 4510 | 4208 | +302 | MM（HYDROGEL/VFE/VEV_4000） |
| Mark 38 | 2493 | 2507 | -14 | MM（HYDROGEL/VEV_4000） |
| Mark 55 | 3254 | 3297 | -43 | MM（仅 VFE） |
| Mark 49 | 115 | 1071 | -956 | 小卖家（VFE） |
| **Mark 67** | 1510 | **0** | **+1510** | **纯单边买家**（仅 VFE，永不卖） |

特殊：**Mark 01 和 Mark 22 在 VEV_6000/6500 上完全配对（1105/1105）**——他们直接互相对冲。

### Follow-through 分析（50 ticks 后 Δmid，3 天合并）

| Actor 行为 | Δmid_+50 含义 | 解读 |
|---|---|---|
| Mark 67 buy VFE | +1.92 | 价格涨 → smart buyer，跟 |
| Mark 49 sell VFE | mid +1.99 | 价格涨 → dumb seller，反向（买） |
| Mark 22 sell VFE | mid +1.87 | 价格涨 → dumb seller，反向 |
| Mark 14 sell HYDROGEL | -1.14 | 价格跌 → smart seller，跟 |
| Mark 38 buy HYDROGEL | mid -1.12 | 价格跌 → dumb buyer，反向（卖） |
| Mark 22 sell VEV_5200 | -0.77 | 弱信号 |

### Cross-day robustness（event study, day1+2 vs day3）

⚠️ **0 个信号同时满足 sign-consistent + |t_oos|≥2**。

最强一致信号（按 |t|）：
- HYDROGEL Mark 14 sell: in -1.06 / out -1.32 (t=-1.23)
- HYDROGEL Mark 38 buy:  in -1.07 / out -1.21 (t=-1.14)
- VFE Mark 67 buy:  in +2.27 / out +1.01 ← alpha 减半到 day3
- VFE Mark 49 sell: in +2.49 / out +0.93 ← 减 63%
- **VFE Mark 22 sell: in +2.43 / out +0.08 ← day3 alpha 完全消失**

**结论**：单 actor follow-through 信号 magnitude 在 1-2 价位，**而 spread 是 6-10**，任何 cross-spread 用法都是 -EV。

---

## 3. 关键架构决策（哪些有效 / 无效）

### ✅ 有效（保留）

| 机制 | 来源 | 价值 |
|---|---|---|
| **Warmup-freeze fair**（短窗口 N=300 估 trimmed mean 后冻结） | 替代 hardcode 的唯一可行办法 | 基础架构 |
| **Drift correction**（freeze 后累计 cum_mean，与 frozen 按 ramp 权重 blend） | 应对日内/跨日漂移 | 必需 |
| **STD_INFLATE = 1.70**（短 warmup 的 std 偏低，乘 1.7 补回） | 经验 | 关键 |
| **Microprice 替代 mid**（top-of-book 加权） | R3 经验 | +18K backtest |
| **Heavy/panic thresh 0.55 / 0.70** | R3 sweep 最优 | +32K backtest |
| **5-layer passive MM, 前重 weights `[0.30,0.25,0.20,0.15,0.10]`** | R3 实盘修复 | 减 spread leak |
| **TAKE_T = 1.0σ, REV_T = 2.25σ** | R3 sweep | 核心 alpha 来源 |
| **I_top 过滤 ±0.5**（强逆向 imbalance 屏蔽 take） | R3 防逆选 | 中性，live 保护 |
| **VEV_6000/6500 → unwind only**（std=0 死产品） | 数据观测 | 0 影响但安全 |
| **Counterparty defensive take blocker (Variant D)** | event study | +7.7K |
| **Ridge ML inventory target bias** (TARGET_SCALE=0.15) | colab pipeline | +10K |

### ❌ 无效（不要再做）

| 机制 | 失败原因 |
|---|---|
| Counterparty Variant A (fair offset, ±1.5) | 信号 ~5% ticks 触发，offset 累积成持续偏置 → -173K |
| Counterparty Variant B (independent take, size 20) | Cross spread cost 6-10 > alpha 1-2 → -883K |
| Counterparty Variant B (size 2-5) | 仍 -150K，spread cost 数学上抵不过 |
| Counterparty Variant C (passive layer skew) | Layers 在 normal regime 几乎不 fill，size 调了也无效 |
| Ridge fair offset (forecast → fair shift) | -195K，spread cost 增加 > alpha |
| Ridge take filter (forecast 强则 block adverse take) | -170K，破坏 MR 短 horizon alpha |
| Adverse-selection ML (预测下 1 tick smart Mark 行为) | AUC 全部 ≈ 0.50（随机），lift 0.6-1.0x |
| EWMA / α=0.0005 cumulative mean 作为 fair | "EWMA 陷阱"：fair 跟价格漂 → 杀 MR alpha |
| Butterfly arb on B(K)=mid(K-100)-2mid(K)+mid(K+100) | -605K disaster（与 per-voucher MR 抢 alpha） |
| Voucher fair = `VFE_frozen - K`（intrinsic anchor） | bias 跨产品全相关化，比 own-frozen 差 |
| 全局 voucher cross-strike position cap | 越 cap 越亏（R3 实测）|
| ML 对 mid_+200 直接预测 LightGBM 部署 | 沙箱无 lightgbm，且 lightgbm cross-day 全部 R²<0 |

### 🟡 中性（无明显效果但保留架构）

- VEV_4500 走 _trade_mr 而非 IV smile：等价
- Counterparty Variant C (passive size skew with Ridge)：~neutral

---

## 4. ML pipeline 主要发现

### Pipeline（colab_round4_counterparty.ipynb）

1. Cell 1-2: 加载 R4 csv
2. Cell 3-4: 大规模 feature engineering（~200 features per product）
   - microstructure: I_top, microprice deviation, spread, returns at 6 lags, rvol_20/100
   - counterparty: per-actor `b_X`/`s_X` per tick + rolling `nf_X_W{5,20,50,200}` + `cum_X`（cumulative inventory）+ pair interaction
   - cross-product: voucher 加 vfe_mid, intrinsic, time_premium, tp_roll, tp_dev
3. Cell 5: LightGBM with leave-day-out CV
4. Cell 6: SHAP top features
5. Cell 7: Conditional alpha by quantile
6. Cell 8: Adverse-selection classification (AUC 全 ≈0.5, 失败)
7. Cell 9: Distill ranked rule table (1241 rules, max |t|=77)
8. Cell 11: Ridge regression distillation per (product, k=50/200)
9. Cell 12: Export `RIDGE_MODELS` dict

### 关键 Ridge 结果（k=200, OOS sharpe）

| Product | Ridge sharpe | LightGBM sharpe | top features |
|---|---|---|---|
| VEV_4000 | **+5.20** | +1.38 | vfe_mid, cum_Mark 14, cum_Mark 38, nf_Mark 14_W200 |
| VEV_5300 | **+5.03** | +0.56 | tp_roll, time_premium, cum_Mark 01, cum_Mark 22 |
| VEV_4500 | **+4.82** | +2.37 | vfe_mid, tp_roll, rvol_100, cum_Mark 22 |
| VEV_5100 | +2.35 | +1.35 | vfe_mid, tp_roll, tp_dev |
| VEV_5000 | +2.34 | +2.18 | tp_roll, vfe_mid, rvol_100 |
| VEV_5200 | +1.99 | +2.89 | cum_Mark 14, tp_roll, vfe_mid |
| VFE | +1.44 | +0.70 | cum_Mark 01, cum_Mark 14, cum_Mark 67 |
| HYDROGEL | +0.74 | -1.16 | cum_Mark 38, cum_Mark 22, cum_Mark 14 |

**核心信号**：
- `cum_Mark_X`（actor 累计净仓位）是最强 alpha 来源（不是 point-in-time）
- `vfe_mid` 对所有 voucher 都重要（underlying 锚）
- `tp_roll`（rolling time_premium 均值）是 voucher 的 fair anchor

### Ridge 部署的 3 种方式 vs PnL

| 方式 | 最优 backtest | 解读 |
|---|---|---|
| Fair offset (forecast → fair shift) | -195K vs base | spread cost 抵消 alpha |
| Take filter (block adverse take when strong forecast) | -170K vs base | 牺牲 MR 短 horizon |
| **Inventory target bias**（target_pos = forecast/std × scale） | **+10K vs D** | 唯一可行整合 |

**为什么 sharpe 5 ≠ 5x PnL**：Ridge 是 200-tick horizon 预测，MR 是 5-50 tick 操作，两者 horizon 不匹配。Inventory target 是用 Ridge 推荐的"中性位"让 MR 围绕它操作，互不干扰。理论 ceiling ≈ 60K/day per product × 8 products ≈ 270K/day total。**900K/day 数学不可达**。

---

## 5. 当前最优参数（v4, backtest 472K）

```python
# Warmup
WARMUP_FREEZE_N = 300         # MUST << 1000 (live tick budget)
DRIFT_CORRECT_START = 400
TRIM_Q = 0.05
STD_INFLATE = 1.70

# Drift correction
DRIFT_RAMP_N = 300
DRIFT_MAX_W = 0.8
DRIFT_MAX_W_STD = 0.7
DRIFT_GATE = 0.0              # always blend after start, no gate

# Trading
TAKE_T_MULT = 1.00            # take when |dev| >= 1σ
REV_T_MULT = 2.25
HEAVY_FRAC = 0.55
PANIC_FRAC = 0.70
PER_SIDE_FRAC = 0.50
LAYER_QTY_WEIGHTS = (0.30, 0.25, 0.20, 0.15, 0.10)
LAYER_WIDTHS_SIGMA = (0.15, 0.30, 0.50, 0.80, 0.80)
I_TOP_BLOCK = 0.5

# Counterparty defensive blocker
BLOCK_BUY_TAKE_RULES = {
    "HYDROGEL_PACK": [("Mark 14", "seller"), ("Mark 38", "buyer")]
}
BLOCK_SELL_TAKE_RULES = {
    "VELVETFRUIT_EXTRACT": [("Mark 67", "buyer"), ("Mark 49", "seller"),
                            ("Mark 22", "seller"), ("Mark 14", "seller")]
}

# Ridge ML
RIDGE_TARGET_SCALE = 0.15     # forecast/std → target_pos fraction
RIDGE_TARGET_MAX_FRAC = 0.30  # cap target_pos at 30% of limit
RIDGE_FILTER_THRESH = 0.0     # filter disabled
RIDGE_WARMUP_TICKS = 100      # need 100 ticks of feature buffer

# Skip dead products
SKIP_PRODUCTS = {"VEV_6000", "VEV_6500"}
```

---

## 6. 重写时一定要避免的实现细节坑

### A. Voucher fair anchor 不能用 VFE microprice live 值

`fair = VFE_microprice - K` 听起来很美，但 voucher mid 必须 mean-revert 到 STATIC level，不是动态值。
- ✅ 正确: `fair = VFE_frozen_mean - K + frozen_tp` （frozen 静态值）
- ❌ 错误: `fair = VFE_current_microprice - K` → 失去 MR anchor → -200K

### B. Feature 对齐（match training shift(1)）

训练时：
- `nf_X_W200 = net.rolling(200).sum().shift(1)` ← 排除当前 tick
- `cum_X = net.cumsum().shift(1)` ← 排除当前
- `ret_K = mid.diff(K)` ← **包括** 当前
- `tp_roll = tp.rolling(200).mean().shift(1)` ← 排除当前

部署时：
- 必须先 update `mids`/`tp` 到当前 tick → 再 compute features → 再 update `cum`/`nf_buf`
- 不然要么 mid features 用了过去的（ret 错），要么 cum/nf 用了当前的（off by 1）

### C. Counterparty 信号衰减必须有上界

如果信号每 5% ticks 触发，decay 0.97 per tick → steady state offset = `delta / (1 - 0.97^20)` ≈ 3.3。
- 必须 clamp 到 [-1, +1] 防止累积爆炸
- 单 tick 内同 actor/role 多次触发要 dedup（不要叠加）

### D. traderData 大小限制

每 product 维护：
- mids buffer (≤260 floats) ~2.6KB
- tp buffer (≤260) ~2.6KB
- nf_buf (7 actors × ≤200 ints) ~4KB
- cum (7 floats) ~150 bytes

12 products → ~110KB。如果 prosperity 沙箱有 traderData 大小限制（典型 100-200KB），可能爆。
- 如果 PnL=0 不是 warmup 问题，检查 traderData
- 优化：ROLL_W_LONG=100 而非 200（buffer 减半），或仅维护有 Ridge model 的产品

### E. Ridge 模型加载

每个产品的 ridge model 是 8-feature 线性回归。`features` 名称必须严格匹配 `_ridge_compute_features` 的输出 key。注意：
- Mark 名字带空格："Mark 14" 不是 "Mark14"
- feature key 格式: `cum_Mark 14`, `nf_Mark 14_W200`（中间 underscore，actor 内空格）
- 缺任何一个 feature → forecast 返回 None → 不应用（不要崩）

### F. 复制-粘贴的 import 顺序

正确：
```python
from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Dict
import json
```
不能加 `import os`（沙箱 forbidden）。也别加 numpy/pandas（不一定有）。

---

## 7. Backtest baselines（R4 数据，3 day total）

| Trader | WARMUP | Backtest PnL | Live ready? |
|---|---|---|---|
| trader_round3.py（hardcode） | (R3 logic) | 667,940 | ❌ no（hardcode 违规 + WARMUP=2000） |
| trader_round4_base.py（无 hardcode, WARMUP=1500） | 1500 | 505,952 | ❌ no（live=1000<1500，PnL=0） |
| trader_round4_D.py（+counterparty defensive）| 1500 | 513,694 | ❌ no |
| trader_round4_v4.py（+Ridge target，旧版） | 1500 | 524,485 | ❌ no |
| **trader_round4_base.py（修正 WARMUP=300）** | 300 | **464,078** | ✅ |
| **trader_round4_D.py（修正 WARMUP=300）** | 300 | **470,618** | ✅ |
| **trader_round4_v4.py（修正 WARMUP=300）** | 300 | **472,158** | ✅ |

---

## 8. 提交前自检 checklist

```bash
# 1. 没有 import os（提交 reject）
grep -E "^import os|os\." trader_round4_*.py
# 期望 0 行

# 2. WARMUP 必须 << 1000
grep -E "WARMUP_FREEZE_N|FREEZE_N|SESSION_FREEZE" trader_round4_*.py
# 任何 ≥ 800 都 risky；推荐 ≤ 500

# 3. 没有 future function
# 手动检查所有 .shift(-N) / lookahead 索引

# 4. 语法检查
python3 -c "import ast; ast.parse(open('trader_round4_v4.py').read()); print('OK')"

# 5. 跑一遍 backtest 确认 PnL 正常
python3 run_bt.py trader_round4_v4.py 4 --no-progress --data bt_data
```

---

## 9. 文件清单

| 文件 | 用途 |
|---|---|
| `trader_round4_v4.py` | 推荐提交版本 |
| `trader_round4_D.py` | 备用（无 Ridge） |
| `trader_round4_base.py` | 兜底（无 Ridge + 无 counterparty） |
| `round4_ridge_models.py` | Ridge 权重 dict（已 inline 进 v4） |
| `round4_ridge_models.json` | 同上 JSON 格式 |
| `round4_alpha_rules.csv` | 1241 条 ML 蒸馏规则（参考） |
| `colab_round4_counterparty.ipynb` | ML pipeline（colab） |
| `ml_counterparty_round4.py` | 本地 ML 探索脚本 |
| `ml_event_study.py` | counterparty cross-day 验证 |
| `bt_data/round4/*.csv` | R4 backtest 数据 |
| `ROUND_4/*.csv` | 原始 R4 数据 |

---

## 10. 重写优先级建议

如果完全重写，按以下顺序最稳：

1. **基础架构**：warmup-freeze + drift correction + 5-layer MM around fair（最关键，~ 90% PnL 来源）
2. **STD_INFLATE = 1.70** + **DRIFT_MAX_W = 0.8** + **WARMUP=300**（live 必须）
3. **Counterparty defensive block**（小 alpha 但稳）
4. **Ridge ML inventory target**（最后加，要正确处理 feature alignment 和 traderData 大小）

每加一层后都要 backtest 确认不退步。建议每加一个机制就 commit。
