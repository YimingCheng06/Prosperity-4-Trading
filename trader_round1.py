from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
try:
    import jsonpickle
except ImportError:
    jsonpickle = None


class Trader:

    POSITION_LIMIT = 80
    ASH_CENTER = 10000

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

    def trade_ash(self, order_depth: OrderDepth, position: int, product: str) -> List[Order]:
        """
        ASH_COATED_OSMIUM:
        以 10000 为长期均值，结合 L1 imbalance、microprice 和库存做动态均值回归做市。
        """
        orders: List[Order] = []

        if not order_depth.buy_orders or not order_depth.sell_orders:
            return orders

        best_bid = max(order_depth.buy_orders.keys())
        best_ask = min(order_depth.sell_orders.keys())

        bid_vol = abs(order_depth.buy_orders[best_bid])
        ask_vol = abs(order_depth.sell_orders[best_ask])

        mid = (best_bid + best_ask) / 2

        total_top = bid_vol + ask_vol
        if total_top > 0:
            imbalance = (bid_vol - ask_vol) / total_top
            microprice = (best_ask * bid_vol + best_bid * ask_vol) / total_top
        else:
            imbalance = 0.0
            microprice = mid

        reservation = (
            self.ASH_CENTER
            + 0.15 * (self.ASH_CENTER - mid)
            + 3.0 * imbalance
            - 0.8 * (microprice - mid)
            - 0.05 * position
        )

        buy_limit = self.POSITION_LIMIT - position
        sell_limit = self.POSITION_LIMIT + position

        buy_taken = 0
        sell_taken = 0

        # 只吃最优一档，减少过度 aggressive
        take_buy_threshold = reservation - 0.5
        take_sell_threshold = reservation + 0.75

        if best_ask <= take_buy_threshold and buy_limit > 0:
            qty = min(abs(order_depth.sell_orders[best_ask]), buy_limit)
            if qty > 0:
                orders.append(Order(product, best_ask, qty))
                buy_taken += qty

        if best_bid >= take_sell_threshold and sell_limit > 0:
            qty = min(abs(order_depth.buy_orders[best_bid]), sell_limit)
            if qty > 0:
                orders.append(Order(product, best_bid, -qty))
                sell_taken += qty

        remaining_buy = buy_limit - buy_taken
        remaining_sell = sell_limit - sell_taken

        # 超过阈值后不再继续向同一方向加仓
        post_buy = position < 55
        post_sell = position > -55

        merged: dict[int, int] = {}

        def add_order(price: int, qty: int) -> None:
            if qty == 0:
                return
            merged[price] = merged.get(price, 0) + qty

        if post_buy and remaining_buy > 0:
            for off, size in [(1, 15), (2, 25)]:
                if remaining_buy <= 0:
                    break
                px = int(round(reservation - off))
                px = max(px, best_bid + 1)
                px = min(px, best_ask - 1)
                qty = min(size, remaining_buy)
                add_order(px, qty)
                remaining_buy -= qty

        if post_sell and remaining_sell > 0:
            for off, size in [(1, 15), (2, 25)]:
                if remaining_sell <= 0:
                    break
                px = int(round(reservation + off))
                px = min(px, best_ask - 1)
                px = max(px, best_bid + 1)
                qty = min(size, remaining_sell)
                add_order(px, -qty)
                remaining_sell -= qty

        for price, qty in merged.items():
            if qty != 0:
                orders.append(Order(product, price, qty))

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

        taken = 0
        if order_depth.sell_orders:
            for price in sorted(order_depth.sell_orders.keys()):
                if buy_limit - taken > 0:
                    vol = abs(order_depth.sell_orders[price])
                    qty = min(vol, buy_limit - taken)
                    orders.append(Order(product, price, qty))
                    taken += qty

        remaining = buy_limit - taken
        if remaining > 0:
            if best_ask is not None:
                orders.append(Order(product, best_ask + 1, remaining))
            elif best_bid is not None:
                orders.append(Order(product, best_bid + 1, remaining))

        return orders