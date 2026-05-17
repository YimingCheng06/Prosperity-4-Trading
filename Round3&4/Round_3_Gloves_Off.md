# Round 3 — "Gloves Off"

欢迎来到 **Solvenar**!这是一颗以技术创新、强劲经济和繁荣文化产业著称的高度发达星球。这个令人惊叹的社会将成为 **Great Orbital Ascension Trials(GOAT)** 的舞台。在这场 **Great Galactic Trade-Off** 中,你将与其他交易团队正面交锋,争夺"银河交易冠军"(Trading Champion of the Galaxy)的称号。

本轮是 GOAT 的开端,**所有团队的 PnL 都从零开始,排行榜也会被重置**。

你需要开发一个新的 Python 程序,纳入针对以下三类产品的交易策略:

- **Hydrogel Packs**(`HYDROGEL_PACK`)
- **Velvetfruit Extract**(`VELVETFRUIT_EXTRACT`)
- **10 个 Velvetfruit Extract Vouchers**(`VELVETFRUIT_EXTRACT_VOUCHER`)——这些 voucher 赋予你在未来某个时间点以特定行权价(strike price)购买 Velvetfruit Extract 的权利。

为了拉开 GOAT 的序幕,**Celestial Gardeners' Guild**(天界园丁公会)将罕见地登场,提供向他们购买 **Ornamental Bio-Pods** 的机会。你可以提交两个报价(offers),并按照你的策略与尽可能多的"Guardeners"进行交易以最大化利润。一旦你成功获得 Bio-Pods,它们将在下一轮交易开始前自动转化为利润。

> ⚠️ **注意:Solvenar 上的每个交易回合(Solvenarian days)只持续 48 小时。**

请果断、彻底、迅速地行动,让通往最终冠军头衔的第一步算数。

---

## 回合目标(Round Objective)

1. 创建一个新的 Python 程序,代表你算法化地交易 `HYDROGEL_PACK`、`VELVETFRUIT_EXTRACT` 以及 `VELVETFRUIT_EXTRACT_VOUCHER`,并在这一最终阶段中产生你的第一笔利润。
2. 此外,手动提交两个报价,与 Celestial Gardeners' Guild 的成员交易 Ornamental Bio-Pods,然后自动出售你获得的 Bio-Pods 以产生额外利润。

---

## 算法交易挑战:"Options Require Decisions"

你交易的三类产品中包含 **2 个资产类别(asset classes)**:

- `HYDROGEL_PACK` 和 `VELVETFRUIT_EXTRACT` 属于 **"delta 1" 产品**,与教程及 Round 1、Round 2 中的产品类似。
- 10 个 `VELVETFRUIT_EXTRACT_VOUCHER` 产品(每个有不同的行权价)属于 **期权(options)**,因此遵循不同的动态规律。

所有产品都是独立交易的,即使由于期权特性,`VELVETFRUIT_EXTRACT_VOUCHER` 的价格可能与 `VELVETFRUIT_EXTRACT` 相关。

### Voucher 列表

Vouchers 命名为:

`VEV_4000`、`VEV_4500`、`VEV_5000`、`VEV_5100`、`VEV_5200`、`VEV_5300`、`VEV_5400`、`VEV_5500`、`VEV_6000`、`VEV_6500`

其中:

- **VEV** = **V**elvetfruit **E**xtract **V**oucher
- 数字代表 **行权价(strike price)**

### 到期时间(Time Till Expiry, TTE)

所有 voucher 的到期期限为 **7 天**,从 Round 1 开始计算,每个回合代表 1 天:

| 时间节点 | TTE |
|---|---|
| Round 1 开始时 | 7 天 |
| Round 2 开始时 | 6 天 |
| Round 3 开始时 | 5 天 |
| ……以此类推 | …… |

### 持仓限制(Position Limits)

(关于上下文和故障排查,请参见 Position Limits 页面)

| 产品 | 持仓限制 |
|---|---|
| `HYDROGEL_PACK` | 200 |
| `VELVETFRUIT_EXTRACT` | 200 |
| `VELVETFRUIT_EXTRACT_VOUCHER`(10 个 voucher 中每一个) | 300 |

### 📃 示例:`VEV_5000`

`VEV_5000` 是基于标的 VEV 的期权,行权价为 5000,持仓限制为 300。

在 Round 3 最终模拟开始时,其到期时间(TTE)为 5 天。在历史数据中,对应的 TTE 值为:

- 历史 day 0(对应 tutorial round)开始时:**TTE = 8 天**
- 历史 day 1(对应 Round 1)开始时:**TTE = 7 天**
- 历史 day 2(对应 Round 2)开始时:**TTE = 6 天**

### 其他规则

- Vouchers **不能在到期前行使**。
- 库存 **不会** 结转到下一回合。
- 与之前的回合一样,任何未平仓位将在回合结束时按 **隐藏公允价值(hidden fair value)** 自动平仓。

---

## 手动交易挑战:"The Celestial Gardeners' Guild"

你需要与 **数量保密** 的对手方进行交易,他们都有一个介于 **670 与 920** 之间的保留价(reserve price)。你与每个对手方最多交易一次。在第二天的交易日,你能够以 **920** 的公允价格出售所有产品。

### Bid 的分布

Bid 的分布在 **670 到 920(两端均包含)** 之间,以 **5 为单位均匀分布**(uniformly distributed in increments of 5)。

> 📃 **示例**:对手方的保留价可能是 675 或 680,但**不会**是 676、677、678、679 等。

### 交易规则

你需要提交 **两个 bid**(bid1 和 bid2):

- **第一个 bid(bid1)**:如果 bid1 **高于** 对手方的保留价,他们会以你的第一个 bid 与你交易。
- **第二个 bid(bid2)**:
  - 如果 bid2 **高于** 对手方的保留价,**且** **高于** 所有玩家第二个 bid 的均值(`avg_b2`),你以第二个 bid 成交。
  - 如果 bid2 **高于** 对手方的保留价,但 **低于或等于** 所有玩家第二个 bid 的均值,成交概率会**急剧下降**:你仍以第二个 bid 成交,但 **PnL 会被惩罚**,惩罚系数为:

$$
\left( \frac{920 - b_2}{920 - \text{avg\_b2}} \right)^3
$$

### 提交订单

在 **Manual Challenge Overview** 窗口中直接提交你的两个 bid,然后点击 **"Submit"** 按钮。你可以在交易回合结束前重新提交新的 bid。回合结束时,**最后提交的 bid** 将被提供给 Celestial Gardeners' Guild 的成员。

---

**来源**:[Round 3 - "Gloves Off" — IMC Prosperity Notion 页面](https://imc-prosperity.notion.site/Round-3-Gloves-Off-34ce8453a0938072a58cc7de372ff551)
