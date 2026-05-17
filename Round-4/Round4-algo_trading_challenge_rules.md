# Algorithmic Trading Challenge: "Hello, I'm Mark"

> Round 4 - 算法交易挑战完整规则

---

## 一、挑战概述

本轮算法交易部分延续 Round 3 的产品体系,但引入了**对手方信息透明化**这一关键变化。你现在可以识别市场中的每一个其他参与者,并研究他们的交易行为。

**Round 4 的整体背景**:
- Frontier Trade Watch (FTW) 披露了市场中活跃对手方的信息
- 对手方 ID 已被添加到 Data Capsule 中的历史交易数据
- 题目原文提示:理解对手方的交易行为和它们带来的独特机会,可能会让"懂得分辨利润与虚张声势"的团队获得优势

---

## 二、可交易产品

产品与 Round 3 相同:

| 产品 | 符号 |
|------|------|
| Hydrogel Packs | `HYDROGEL_PACK` |
| Velvetfruit Extract | `VELVETFRUIT_EXTRACT` |
| 10 Velvetfruit Extract Vouchers | `VELVETFRUIT_EXTRACT_VOUCHER` |

---

## 三、本轮关键变化:对手方信息

### `Trade` 类的变化

数据模型中定义的 `Trade` 类(详见 Appendix B: datamodel.py):

```python
class Trade:
    def __init__(self, symbol: Symbol, price: int, quantity: int, buyer: UserId =
        self.symbol = symbol
        self.price: int = price
        self.quantity: int = quantity
        self.buyer = buyer
        self.seller = seller
        self.timestamp = timestamp

        # Some methods
```

### 字段含义对比

| 轮次 | `self.buyer` / `self.seller` |
|------|------------------------------|
| Round 1, 2, 3 | 始终为 `None`(无对手方信息) |
| **Round 4** | **代表参与者的真实名称** |

### 如何利用这一信息

题目明确鼓励:**"Please feel free to leverage this information however you see fit, and refine your strategy using this extra visibility!"**

即:可以自由使用这些信息来优化策略,例如:
- 识别每个对手方的行为模式
- 区分"有信息优势"与"信息劣势"的参与者
- 针对不同对手方采取差异化定价或下单策略

---

## 四、持仓限额

持仓限额(position limits)与 Round 3 保持不变:

| 产品 | 限额 |
|------|------|
| `HYDROGEL_PACK` | 200 |
| `VELVETFRUIT_EXTRACT` | 200 |
| `VELVETFRUIT_EXTRACT_VOUCHER` | 每个代金券 300 |

> 关于持仓限额的额外背景与故障排查,参见 Position Limits 页面。

---

## 五、Voucher(代金券期权)说明

### 示例
基于 Round 3 的示例延续:

> `VEV_5000` 是一个行权价为 5000 的期权,在 Round 4 中 **TTE = 4 天**,持仓上限为 300。

### 含义
- 共有 10 个 Voucher(对应不同行权价的期权)
- 每个 Voucher 的 TTE(Time To Expiry,到期时间)在 Round 4 中已确定
- 每个 Voucher 独立计算 300 的持仓上限

---

## 六、Round 目标

> Optimize your Python program to trade `HYDROGEL_PACK`, `VELVETFRUIT_EXTRACT`, and `VELVETFRUIT_EXTRACT_VOUCHER`, incorporating the newly disclosed counterparty information into your strategy.

**核心任务**:
1. 优化你的 Python 交易程序
2. 交易上述三类产品
3. **将新披露的对手方信息融入策略**

---

## 七、与手动交易部分的关系

题目原文明确说明:

> "Be aware that these exotic options operate independently from your algorithmic trading activities."

即:
- 算法交易(本部分)与手动交易(Aether Crystal 及其期权)**完全独立运行**
- 两部分的盈亏分别计算
- 两部分的策略可以完全分离设计

---

## 八、关键要点速查

| 要点 | 说明 |
|------|------|
| 产品同 Round 3 | HYDROGEL_PACK / VELVETFRUIT_EXTRACT / 10 个 VOUCHER |
| 新增信息 | `Trade.buyer` 和 `Trade.seller` 现在是真实参与者名称 |
| 历史数据 | Data Capsule 中的历史交易数据已附带对手方 ID |
| 持仓限额 | 200 / 200 / 300(每个 Voucher) |
| Voucher 示例 | `VEV_5000`:行权价 5000,Round 4 中 TTE=4 天 |
| 与手动部分 | 完全独立 |

---

## 九、原文要点引用

- **市场透明度**:"you can identify every other participant in the market and study their behavior."
- **字段语义变化**:"these `self.buyer` and `self.seller` fields now represent the names of the participants!"
- **使用自由度**:"Please feel free to leverage this information however you see fit, and refine your strategy using this extra visibility!"
- **独立性**:"these exotic options operate independently from your algorithmic trading activities."
