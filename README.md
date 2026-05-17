# Prosperity Round 5 — Algorithmic Trading

IMC Prosperity "The Final Stretch" 第五轮算法交易方案。

本轮共 50 个全新可交易商品，均匀分为 10 个类别（每类 5 个），每个商品持仓上限为 10。
挑战主题为 **"Cherry Picking Winners"**：部分类别藏有可被利用的强模式与市场低效，
其余类别则采用稳健的做市策略兜底。

## 策略概览

交易逻辑集中在单文件 `trader_baseline.py`，按类别分层：

| 层级 | 类别 | 策略 |
|------|------|------|
| Tier 1 (GOLD) | Purification Pebbles | 线性守恒 `XS+S+M+L+XL == 50000`，残差条件被动做市 |
| Tier 1 (GOLD) | Protein Snack Packs | 两组配对篮子（CHOC+VAN、STRAW+RASP）z-score 均值回归 |
| Tier 2 (SILVER) | Vertical Sleeping Pods | POLY≈0.964·COTTON、LAMB≈0.401·NYLON 配对回归 |
| Tier 2 (SILVER) | Domestic Robots | DISHES 对其余 4 个等权篮子 |
| Tier 2 (SILVER) | UV-Visors | AMBER 与 MAGENTA 配对回归 |
| Tier 3 (BRONZE) | Galaxy Sounds / Microchips / Translators / Panels | 小仓位配对回归（`|pos| ≤ 3`） |
| Tier 4 (NOPE) | Oxygen Shakes | 跳过（hedge ratio 跨日变号，无稳定结构） |

未进入主策略的商品由全市场被动做市层（universal MM）兜底，并按 bid/ask
imbalance 做 microprice 偏移报价。

## 文件说明

| 文件 | 说明 |
|------|------|
| `trader_baseline.py` | 提交到 Prosperity 平台的单文件交易程序 |
| `TUNABLE_PARAMS.md` | 全部可调超参清单、网格范围与 hedge ratio 标注 |
| `backtest_round5.ipynb` | Colab 回测 notebook（参数网格搜索、OLS 拟合） |
| `backtest_round5_colab.pdf` | 回测 notebook 的 Colab 运行结果导出 |
| `smoke_test.py` | 用真实数据驱动 `Trader.run` 的烟雾测试，校验无异常、无持仓违例 |
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
