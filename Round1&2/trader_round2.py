from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
try:
    import jsonpickle
except ImportError:
    jsonpickle = None


class Trader:

    POSITION_LIMIT = 80

    def run(self, state: TradingState) -> tuple[dict[str, list[Order]], int, str]:
        result = {}

        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            position = state.position.get(product, 0)

            if product == "ASH_COATED_OSMIUM":
                orders = self.trade_ash(order_depth, position, product)
            elif product == "INTARIAN_PEPPER_ROOT":
                orders = self.trade_pepper(order_depth, position, product)
            else:
                orders = []

            result[product] = orders

        trader_data = ""
        conversions = 0
        return result, conversions, trader_data

    FAIR_CENTER = 10000  # ASH 三天统计均值，std≈5

    def trade_ash(self, order_depth: OrderDepth, position: int, product: str) -> List[Order]:
        """ASH: 5层做市 + 反转偏置 + 反转主动吃单。
        - 基础: fair±(1,2,3,5,7) 5层 posting（锚定 fc=10000）
        - 反转: mid 高于 fc 时，ask 收紧 + bid 外推；mid 低时反之
        - 跨价吃: 偏离 ≥4 且仓位方向对时，主动吃对面盘
        """
        orders: List[Order] = []
        fc = self.FAIR_CENTER

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None
        if best_ask is None and best_bid is None:
            return orders

        if best_bid is not None and best_ask is not None:
            mid = (best_bid + best_ask) / 2
        elif best_bid is not None:
            mid = best_bid + 4
        else:
            mid = best_ask - 4
        deviation = mid - fc

        buy_limit = self.POSITION_LIMIT - position
        sell_limit = self.POSITION_LIMIT + position

        long_heavy = position > 20
        short_heavy = position < -20
        panic_long = position > 60
        panic_short = position < -60

        # === TAKING: 库存感知 ===
        if panic_long:
            take_buy_max, take_sell_min = fc - 3, fc - 1
        elif long_heavy:
            take_buy_max, take_sell_min = fc - 2, fc
        elif panic_short:
            take_buy_max, take_sell_min = fc + 1, fc + 3
        elif short_heavy:
            take_buy_max, take_sell_min = fc, fc + 2
        else:
            take_buy_max, take_sell_min = fc - 1, fc + 1

        buy_done = 0
        sell_done = 0

        if order_depth.sell_orders:
            for price in sorted(order_depth.sell_orders.keys()):
                if price <= take_buy_max and buy_limit - buy_done > 0:
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, buy_limit - buy_done)
                    orders.append(Order(product, price, qty))
                    buy_done += qty
        if order_depth.buy_orders:
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if price >= take_sell_min and sell_limit - sell_done > 0:
                    vol = abs(order_depth.buy_orders[price])
                    qty = min(vol, sell_limit - sell_done)
                    orders.append(Order(product, price, -qty))
                    sell_done += qty

        # === REVERSION TAKE: mid 强偏离且仓位未满时主动跨价建反转仓 ===
        # deviation≥4 → 主动卖（直接到 best_bid，可能比 post 更早被执行）
        # 限制: 总反转仓 ≤ 40（不全 all-in）
        # rev_cap scales with deviation strength: dev=4 → cap 40, dev=8+ → cap 70
        rev_cap = min(70, 20 + int(abs(deviation) * 6))
        if deviation >= 4 and position > -rev_cap:
            rev_room = min(rev_cap + position, sell_limit - sell_done)
            for price in sorted(order_depth.buy_orders.keys(), reverse=True):
                if price >= fc and rev_room > 0:
                    vol = abs(order_depth.buy_orders[price])
                    qty = min(vol, rev_room)
                    if qty > 0:
                        orders.append(Order(product, price, -qty))
                        sell_done += qty
                        rev_room -= qty
        elif deviation <= -4 and position < rev_cap:
            rev_room = min(rev_cap - position, buy_limit - buy_done)
            for price in sorted(order_depth.sell_orders.keys()):
                if price <= fc and rev_room > 0:
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, rev_room)
                    if qty > 0:
                        orders.append(Order(product, price, qty))
                        buy_done += qty
                        rev_room -= qty

        # === POSTING: 5层 + 反转偏置 ===
        # 基础 5 层
        bid_layers = [(fc - 1, 10), (fc - 2, 20), (fc - 3, 20), (fc - 5, 15), (fc - 7, 15)]
        ask_layers = [(fc + 1, 10), (fc + 2, 20), (fc + 3, 20), (fc + 5, 15), (fc + 7, 15)]

        # 反转偏置: mid 高 → ask 内移一档更激进（收更高价），bid 外移一档（别抢多仓）
        if deviation >= 3:
            ask_layers = [(fc, 10), (fc + 1, 20), (fc + 2, 20), (fc + 4, 15), (fc + 6, 15)]
            bid_layers = [(fc - 2, 10), (fc - 3, 20), (fc - 5, 20), (fc - 7, 15)]
        elif deviation <= -3:
            bid_layers = [(fc, 10), (fc - 1, 20), (fc - 2, 20), (fc - 4, 15), (fc - 6, 15)]
            ask_layers = [(fc + 2, 10), (fc + 3, 20), (fc + 5, 20), (fc + 7, 15)]

        # 仓位保护覆盖
        if panic_long:
            ask_layers = [(fc, 20), (fc + 1, 30), (fc + 3, 20)]
            bid_layers = []
        elif long_heavy:
            ask_layers = [(fc + 1, 15), (fc + 2, 30), (fc + 4, 25)]
            bid_layers = []
        elif panic_short:
            bid_layers = [(fc, 20), (fc - 1, 30), (fc - 3, 20)]
            ask_layers = []
        elif short_heavy:
            bid_layers = [(fc - 1, 15), (fc - 2, 30), (fc - 4, 25)]
            ask_layers = []

        remaining_buy = buy_limit - buy_done
        remaining_sell = sell_limit - sell_done

        for price, qty in bid_layers:
            if remaining_buy <= 0:
                break
            q = min(qty, remaining_buy)
            orders.append(Order(product, price, q))
            remaining_buy -= q

        for price, qty in ask_layers:
            if remaining_sell <= 0:
                break
            q = min(qty, remaining_sell)
            orders.append(Order(product, price, -q))
            remaining_sell -= q

        return orders

    def trade_pepper(self, order_depth: OrderDepth, position: int, product: str) -> List[Order]:
        """
        INTARIAN_PEPPER_ROOT: 趋势做多（保底收益）

        简单策略：尽快买满80单位，持有赚趋势。
        不做市，避免卖出损失趋势利润。
        """
        orders: List[Order] = []

        best_ask = min(order_depth.sell_orders.keys()) if order_depth.sell_orders else None
        best_bid = max(order_depth.buy_orders.keys()) if order_depth.buy_orders else None

        buy_limit = self.POSITION_LIMIT - position

        if buy_limit <= 0:
            return orders

        # 吃掉所有卖单
        taken = 0
        if order_depth.sell_orders:
            for price in sorted(order_depth.sell_orders.keys()):
                if buy_limit - taken > 0:
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, buy_limit - taken)
                    orders.append(Order(product, price, qty))
                    taken += qty

        # 挂买单抢剩余
        remaining = buy_limit - taken
        if remaining > 0:
            if best_ask is not None:
                orders.append(Order(product, best_ask + 1, remaining))
            elif best_bid is not None:
                orders.append(Order(product, best_bid + 1, remaining))

        return orders