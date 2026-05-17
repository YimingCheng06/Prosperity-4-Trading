# Round 5 — "The Final Stretch" 规则总结

## 总体概述

最后一轮，最后一次机会冲击排行榜第一名。

- FTW 引入了 **50 个全新的可交易商品**，替代之前几轮的商品。
- **不能再交易往轮的商品**，只能交易这 50 个新商品。
- 50 个商品被均匀划分为 **10 个类别**，每类 5 个。
- 此外，可以在邻近行星 **Ignith** 的市场进行交易，从其 **9 个可交易商品**中挑选作为最后的利润增量。
- Ignith 市场的主要信息源是 **Ashflow Alpha**（新闻源）。

---

## 本轮目标（Round Objective）

1. 编写 **一个最终的 Python 程序**，用于在 10 个类别共 50 个商品中执行交易策略，将其上传到网络，让算法自动产生利润。
2. 利用 **Ashflow Alpha** 新闻源制定策略，从 Ignith 的 **9 个商品**中挑选交易标的，获取最终的利润增量。

---

## 算法交易挑战："Cherry Picking Winners"

- 共 50 个新商品，分为 **10 组，每组 5 个**。
- 每组都有自己的"故事"，部分组中存在 **更明显的市场低效（market inefficiencies）**，价格走势中藏着可被发现的强模式（strong patterns），可以加以利用。
- 对于其他没有明显模式的商品，仍需像之前几轮一样制定有效的交易策略。
- **重要限制**：只能交易下面列出的这 50 个商品，**不能交易往轮商品**。
- **所有商品的持仓上限均为 10**（详见 Position Limits 页面）。

---

## 50 个可交易商品（10 类 × 5 个）

### 1. Galaxy Sounds Recorders（星系声音记录器）
- `GALAXY_SOUNDS_DARK_MATTER`
- `GALAXY_SOUNDS_BLACK_HOLES`
- `GALAXY_SOUNDS_PLANETARY_RINGS`
- `GALAXY_SOUNDS_SOLAR_WINDS`
- `GALAXY_SOUNDS_SOLAR_FLAMES`

### 2. Vertical Sleeping Pods（立式睡眠舱）
- `SLEEP_POD_SUEDE`
- `SLEEP_POD_LAMB_WOOL`
- `SLEEP_POD_POLYESTER`
- `SLEEP_POD_NYLON`
- `SLEEP_POD_COTTON`

### 3. Organic Microchips（有机微芯片）
- `MICROCHIP_CIRCLE`
- `MICROCHIP_OVAL`
- `MICROCHIP_SQUARE`
- `MICROCHIP_RECTANGLE`
- `MICROCHIP_TRIANGLE`

### 4. Purification Pebbles（净化石）
- `PEBBLES_XS`
- `PEBBLES_S`
- `PEBBLES_M`
- `PEBBLES_L`
- `PEBBLES_XL`

### 5. Domestic Robots（家用机器人）
- `ROBOT_VACUUMING`
- `ROBOT_MOPPING`
- `ROBOT_DISHES`
- `ROBOT_LAUNDRY`
- `ROBOT_IRONING`

### 6. UV-Visors（紫外线护目镜）
- `UV_VISOR_YELLOW`
- `UV_VISOR_AMBER`
- `UV_VISOR_ORANGE`
- `UV_VISOR_RED`
- `UV_VISOR_MAGENTA`

### 7. Instant Translators（即时翻译器）
- `TRANSLATOR_SPACE_GRAY`
- `TRANSLATOR_ASTRO_BLACK`
- `TRANSLATOR_ECLIPSE_CHARCOAL`
- `TRANSLATOR_GRAPHITE_MIST`
- `TRANSLATOR_VOID_BLUE`

### 8. Construction Panels（建筑面板）
- `PANEL_1X2`
- `PANEL_2X2`
- `PANEL_1X4`
- `PANEL_2X4`
- `PANEL_4X4`

### 9. Liquid Breath Oxygen Shakes（液态氧气呼吸饮）
- `OXYGEN_SHAKE_MORNING_BREATH`
- `OXYGEN_SHAKE_EVENING_BREATH`
- `OXYGEN_SHAKE_MINT`
- `OXYGEN_SHAKE_CHOCOLATE`
- `OXYGEN_SHAKE_GARLIC`

### 10. Protein Snack Packs（蛋白零食包）
- `SNACKPACK_CHOCOLATE`
- `SNACKPACK_VANILLA`
- `SNACKPACK_PISTACHIO`
- `SNACKPACK_STRAWBERRY`
- `SNACKPACK_RASPBERRY`

---

## 关键约束速查

| 项目 | 数值/说明 |
|------|-----------|
| 可交易商品数（主市场） | 50 个（10 类 × 5） |
| 可交易商品数（Ignith 市场） | 9 个 |
| 每个商品持仓上限 | 10 |
| 是否可交易往轮商品 | **否** |
| 交付物 | 一个最终的 Python 程序 |
| Ignith 市场信息源 | Ashflow Alpha |
