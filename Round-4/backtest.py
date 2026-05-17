"""
Local backtest for trader.py. Reads ROUND_4 CSVs, replays orderbook + market trades,
calls Trader.run(state) per timestamp, simulates fills against the historical book,
and reports PnL by product.

Conservative fill model:
  - Aggressive (taking) orders fill against displayed depth at the snapshot.
  - Passive orders (price <= best_bid for buys, >= best_ask for sells) ASSUMED
    NOT FILLED. This understates strategy PnL but is safe; an optimistic model
    would fill them whenever the book later trades through that level.

Run: python backtest.py
"""

from __future__ import annotations
import csv
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

DATA_DIR = os.path.join(os.path.dirname(__file__), "ROUND_4")
DAYS = [1, 2, 3]


# ---------- Minimal datamodel ----------
@dataclass
class Order:
    symbol: str
    price: int
    quantity: int  # +buy, -sell


@dataclass
class OrderDepth:
    buy_orders: Dict[int, int] = field(default_factory=dict)   # price -> qty (positive)
    sell_orders: Dict[int, int] = field(default_factory=dict)  # price -> qty (negative, IMC convention)


@dataclass
class Trade:
    symbol: str
    price: float
    quantity: int
    buyer: Optional[str]
    seller: Optional[str]
    timestamp: int


@dataclass
class TradingState:
    timestamp: int
    listings: Dict[str, str]
    order_depths: Dict[str, OrderDepth]
    own_trades: Dict[str, List[Trade]]
    market_trades: Dict[str, List[Trade]]
    position: Dict[str, int]
    observations: Dict
    traderData: str


# Inject our datamodel into trader.py's import path
import importlib
import types

dm = types.ModuleType("datamodel")
dm.Order = Order
dm.OrderDepth = OrderDepth
dm.Trade = Trade
dm.TradingState = TradingState
sys.modules["datamodel"] = dm

import trader  # noqa: E402

importlib.reload(trader)  # ensure it picks up our shim


# ---------- Data loading ----------
def load_prices() -> Dict[Tuple[int, int, str], OrderDepth]:
    """Returns {(day, ts, symbol): OrderDepth}."""
    out: Dict[Tuple[int, int, str], OrderDepth] = {}
    for d in DAYS:
        path = os.path.join(DATA_DIR, f"prices_round_4_day_{d}.csv")
        with open(path) as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            for row in reader:
                if not row:
                    continue
                day = int(row[idx["day"]])
                ts = int(row[idx["timestamp"]])
                sym = row[idx["product"]]
                od = OrderDepth()
                for k in (1, 2, 3):
                    bp = row[idx[f"bid_price_{k}"]]
                    bv = row[idx[f"bid_volume_{k}"]]
                    if bp and bv:
                        od.buy_orders[int(bp)] = int(bv)
                    ap = row[idx[f"ask_price_{k}"]]
                    av = row[idx[f"ask_volume_{k}"]]
                    if ap and av:
                        od.sell_orders[int(ap)] = -int(av)
                out[(day, ts, sym)] = od
    return out


def load_trades() -> Dict[Tuple[int, int], List[Trade]]:
    """Returns {(day, ts): [Trade,...]}. Day inferred from file."""
    out: Dict[Tuple[int, int], List[Trade]] = defaultdict(list)
    for d in DAYS:
        path = os.path.join(DATA_DIR, f"trades_round_4_day_{d}.csv")
        with open(path) as f:
            reader = csv.reader(f, delimiter=";")
            header = next(reader)
            idx = {name: i for i, name in enumerate(header)}
            for row in reader:
                if not row:
                    continue
                ts = int(row[idx["timestamp"]])
                t = Trade(
                    symbol=row[idx["symbol"]],
                    price=float(row[idx["price"]]),
                    quantity=int(row[idx["quantity"]]),
                    buyer=row[idx["buyer"]] or None,
                    seller=row[idx["seller"]] or None,
                    timestamp=ts,
                )
                out[(d, ts)].append(t)
    return out


# ---------- Fill simulation ----------
def simulate_fills(orders: List[Order], od: OrderDepth,
                   next_trades: List = None,
                   passive_fill_frac: float = 0.5) -> List[Tuple[int, int, int]]:
    """
    Two-stage fill model:
      1. Aggressive crossing — order price reaches into displayed depth.
      2. Passive — orders that DIDN'T cross (rest on book) fill against the
         next snapshot's printed trades, capped by passive_fill_frac
         (we assume only some fraction of printed counter-flow would have hit
         us instead of the historical taker — conservative).

    Returns list of (price, qty_signed, _) fills.
    """
    fills = []
    asks = sorted(od.sell_orders.items())
    bids = sorted(od.buy_orders.items(), reverse=True)
    asks = [[p, -q] for p, q in asks]
    bids = [[p, q] for p, q in bids]

    best_ask = asks[0][0] if asks else None
    best_bid = bids[0][0] if bids else None

    passive_buys: List[List] = []   # [price, qty_remaining] (positive qty)
    passive_sells: List[List] = []  # [price, qty_remaining] (positive qty)

    for o in orders:
        remaining = o.quantity
        if remaining > 0:  # buy
            crossed = False
            for level in asks:
                if level[1] <= 0:
                    continue
                if o.price < level[0]:
                    break
                crossed = True
                take = min(level[1], remaining)
                if take > 0:
                    fills.append((level[0], take, 0))
                    level[1] -= take
                    remaining -= take
                if remaining <= 0:
                    break
            if remaining > 0 and not crossed and best_ask is not None and o.price < best_ask:
                # Passive bid that rests on book
                passive_buys.append([int(o.price), remaining])
        elif remaining < 0:
            need = -remaining
            crossed = False
            for level in bids:
                if level[1] <= 0:
                    continue
                if o.price > level[0]:
                    break
                crossed = True
                take = min(level[1], need)
                if take > 0:
                    fills.append((level[0], -take, 0))
                    level[1] -= take
                    need -= take
                if need <= 0:
                    break
            if need > 0 and not crossed and best_bid is not None and o.price > best_bid:
                passive_sells.append([int(o.price), need])

    # ----- Passive fills against next-tick printed trades -----
    if next_trades and (passive_buys or passive_sells):
        # Sort our resting orders by price priority (best first)
        passive_buys.sort(key=lambda x: -x[0])
        passive_sells.sort(key=lambda x: x[0])

        for tr in next_trades:
            tp = float(tr.price)
            tq = int(tr.quantity)
            if tq <= 0:
                continue
            avail = max(1, int(tq * passive_fill_frac))
            # Trade printed at tp with qty tq → some seller hit a bid at <= tp,
            # equivalently some buyer hit an ask at >= tp.
            # Our resting bid at price >= tp would have been filled (joined the queue).
            for lvl in passive_buys:
                if avail <= 0:
                    break
                if lvl[1] <= 0:
                    continue
                if lvl[0] >= tp:
                    take = min(lvl[1], avail)
                    fills.append((lvl[0], take, 0))
                    lvl[1] -= take
                    avail -= take
            avail = max(1, int(tq * passive_fill_frac))
            for lvl in passive_sells:
                if avail <= 0:
                    break
                if lvl[1] <= 0:
                    continue
                if lvl[0] <= tp:
                    take = min(lvl[1], avail)
                    fills.append((lvl[0], -take, 0))
                    lvl[1] -= take
                    avail -= take

    return fills


# ---------- Backtest loop ----------
def run_backtest(verbose: bool = True, days: List[int] = None) -> Dict:
    days = days or DAYS
    print(f"Loading data for days {days} ...")
    prices = load_prices()
    trades = load_trades()

    # Build sorted timestamp axis
    timestamps_by_day: Dict[int, List[int]] = defaultdict(set)
    symbols = set()
    for (d, ts, sym) in prices.keys():
        if d not in days:
            continue
        timestamps_by_day[d].add(ts)
        symbols.add(sym)
    timestamps_by_day = {d: sorted(ts) for d, ts in timestamps_by_day.items()}

    t = trader.Trader()
    position: Dict[str, int] = defaultdict(int)
    cash: float = 0.0
    pnl_by_symbol: Dict[str, float] = defaultdict(float)
    fills_log = []
    traderData = ""

    for d in days:
        for ts in timestamps_by_day[d]:
            # Build order_depths snapshot
            ods = {}
            for sym in symbols:
                od = prices.get((d, ts, sym))
                if od is not None:
                    ods[sym] = od
            # market_trades = trades that printed at this ts (info we just observed)
            mts: Dict[str, List[Trade]] = defaultdict(list)
            for tr in trades.get((d, ts), []):
                mts[tr.symbol].append(tr)

            state = TradingState(
                timestamp=ts + (d - 1) * 1_000_000,  # global monotonic ts
                listings={},
                order_depths=ods,
                own_trades={},
                market_trades=mts,
                position=dict(position),
                observations={},
                traderData=traderData,
            )

            try:
                result, _, traderData = t.run(state)
            except Exception as e:
                print(f"[err] day={d} ts={ts} trader.run raised: {e}")
                continue

            # Simulate fills against the same snapshot
            for sym, orders in (result or {}).items():
                if not orders:
                    continue
                od = ods.get(sym)
                if od is None:
                    continue
                fills = simulate_fills(orders, od)
                for px, qty, _ in fills:
                    cash -= px * qty
                    position[sym] += qty
                    pnl_by_symbol[sym]  # init key
                    fills_log.append((d, ts, sym, px, qty))

        # End-of-day: mark to mid for unrealized
        if verbose:
            print(f"\n=== End of day {d} ===")
            mtm = 0.0
            for sym in sorted(symbols):
                pos = position[sym]
                # last available mid that day
                last_ts = timestamps_by_day[d][-1]
                od = prices.get((d, last_ts, sym))
                if od and od.buy_orders and od.sell_orders:
                    mid = 0.5 * (max(od.buy_orders) + min(od.sell_orders))
                else:
                    mid = 0.0
                sym_mtm = pos * mid
                mtm += sym_mtm
                if abs(pos) > 0 or pnl_by_symbol[sym]:
                    print(f"  {sym:25s} pos={pos:+6d}  mid={mid:8.2f}  mtm_value={sym_mtm:+12.2f}")
            print(f"  cash={cash:+.2f}  total_equity={cash + mtm:+.2f}")

    # Final PnL
    final_mtm = 0.0
    last_day = days[-1]
    last_ts = timestamps_by_day[last_day][-1]
    print("\n=== FINAL ===")
    for sym in sorted(symbols):
        pos = position[sym]
        od = prices.get((last_day, last_ts, sym))
        if od and od.buy_orders and od.sell_orders:
            mid = 0.5 * (max(od.buy_orders) + min(od.sell_orders))
        else:
            mid = 0.0
        sym_mtm = pos * mid
        final_mtm += sym_mtm
        # per-symbol cash contribution (approx — we lump cash globally)
        if abs(pos) > 0:
            print(f"  {sym:25s} final_pos={pos:+6d}  mid={mid:8.2f}  mtm={sym_mtm:+12.2f}")

    total = cash + final_mtm
    print(f"\nCash: {cash:+.2f}")
    print(f"MTM:  {final_mtm:+.2f}")
    print(f"PnL:  {total:+.2f}")
    print(f"Fills: {len(fills_log)}")

    return {
        "cash": cash,
        "mtm": final_mtm,
        "total": total,
        "fills": fills_log,
        "final_position": dict(position),
    }


def run_window(day: int, ts_start: int, ts_end: int, prices, trades, verbose: bool = False,
               passive_fill_frac: float = 0.5) -> float:
    """Run trader on [ts_start, ts_end) of one day. Mark-to-mid at end. Return final equity.

    passive_fill_frac controls the optimistic passive-fill model:
      0.0 = no passive fills (super conservative — old behavior).
      0.5 = assume half of next-tick printed counter-flow would have hit our resting orders.
      1.0 = assume our orders would always be queue-priority (very optimistic).
    """
    timestamps = sorted({ts for (d, ts, _) in prices.keys() if d == day and ts_start <= ts < ts_end})
    symbols = sorted({sym for (d, _, sym) in prices.keys() if d == day})

    t = trader.Trader()
    position: Dict[str, int] = defaultdict(int)
    cash: float = 0.0
    traderData = ""

    for i, ts in enumerate(timestamps):
        ods = {sym: prices[(day, ts, sym)] for sym in symbols if (day, ts, sym) in prices}
        mts: Dict[str, List[Trade]] = defaultdict(list)
        for tr in trades.get((day, ts), []):
            mts[tr.symbol].append(tr)
        # Next-tick trades for passive fill simulation
        next_mts: Dict[str, List[Trade]] = defaultdict(list)
        if i + 1 < len(timestamps):
            for tr in trades.get((day, timestamps[i + 1]), []):
                next_mts[tr.symbol].append(tr)

        state = TradingState(
            timestamp=ts,
            listings={},
            order_depths=ods,
            own_trades={},
            market_trades=mts,
            position=dict(position),
            observations={},
            traderData=traderData,
        )
        try:
            result, _, traderData = t.run(state)
        except Exception as e:
            if verbose:
                print(f"  err day={day} ts={ts}: {e}")
            continue
        for sym, orders in (result or {}).items():
            if not orders:
                continue
            od = ods.get(sym)
            if od is None:
                continue
            for px, qty, _ in simulate_fills(orders, od,
                                             next_trades=next_mts.get(sym, []),
                                             passive_fill_frac=passive_fill_frac):
                cash -= px * qty
                position[sym] += qty

    # MTM
    last_ts = timestamps[-1] if timestamps else ts_start
    mtm = 0.0
    for sym in symbols:
        pos = position[sym]
        if pos == 0:
            continue
        od = prices.get((day, last_ts, sym))
        if od and od.buy_orders and od.sell_orders:
            mid = 0.5 * (max(od.buy_orders) + min(od.sell_orders))
            mtm += pos * mid
    eq = cash + mtm
    if verbose:
        print(f"  day={day} window=[{ts_start},{ts_end}) cash={cash:+.0f} mtm={mtm:+.0f} eq={eq:+.0f}")
    return eq


def run_sliding(days: List[int] = None, window: int = 100_000, step: int = 100_000,
                passive_fill_frac: float = 0.5):
    """1 day = 1_000_000 timestamp units. Default window = 1/10 day = 100_000 units = 1000 ticks."""
    days = days or DAYS
    print(f"Loading data for days {days} (passive_fill_frac={passive_fill_frac}) ...")
    prices = load_prices()
    trades = load_trades()
    results = []
    for d in days:
        for start in range(0, 1_000_000, step):
            eq = run_window(d, start, start + window, prices, trades, verbose=True,
                            passive_fill_frac=passive_fill_frac)
            results.append((d, start, eq))
    eqs = [r[2] for r in results]
    n = len(eqs)
    mean = sum(eqs) / n if n else 0.0
    var = sum((x - mean) ** 2 for x in eqs) / n if n else 0.0
    std = var ** 0.5
    sharpe = mean / std if std > 0 else 0.0
    pos = sum(1 for x in eqs if x > 0)
    print(f"\n=== SLIDING SUMMARY ({n} windows) ===")
    print(f"  mean PnL/window: {mean:+.1f}")
    print(f"  std PnL/window:  {std:.1f}")
    print(f"  Sharpe (per win):{sharpe:.2f}")
    print(f"  hit rate:        {pos}/{n} = {pos/n*100:.0f}%")
    print(f"  worst window:    {min(eqs):+.1f}")
    print(f"  best window:     {max(eqs):+.1f}")
    return results


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, nargs="+", default=DAYS)
    ap.add_argument("--quiet", action="store_true")
    ap.add_argument("--mode", choices=["full", "sliding"], default="sliding")
    ap.add_argument("--passive", type=float, default=0.5,
                    help="Passive fill fraction (0=conservative, 0.5=realistic, 1.0=optimistic)")
    args = ap.parse_args()
    if args.mode == "sliding":
        run_sliding(days=args.days, passive_fill_frac=args.passive)
    else:
        run_backtest(verbose=not args.quiet, days=args.days)
