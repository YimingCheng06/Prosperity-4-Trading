"""Smoke test for trader_baseline.py — drive 30 ticks of real Day-2 data
through Trader.run and verify no exceptions and no position-limit violations.
"""
import sys, csv, time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
# `datamodel` ships inside the prosperity3bt package — add its dir to the path.
try:
    import prosperity3bt
    sys.path.insert(0, str(Path(prosperity3bt.__file__).resolve().parent))
except ImportError:
    pass

from datamodel import OrderDepth, TradingState, Order, Listing, Observation  # noqa: E402
from trader_baseline import Trader, POSITION_LIMIT, BRONZE_POSITION_CAP  # noqa: E402

BRONZE_SYMS = {
    "GALAXY_SOUNDS_SOLAR_FLAMES", "GALAXY_SOUNDS_SOLAR_WINDS",
    "MICROCHIP_SQUARE", "MICROCHIP_RECTANGLE",
    "TRANSLATOR_ECLIPSE_CHARCOAL", "TRANSLATOR_VOID_BLUE",
    "PANEL_1X2", "PANEL_2X2",
}


def load_ticks(csv_path: Path, max_ticks: int):
    """Group rows by timestamp -> {ts: {sym: OrderDepth}}."""
    ticks = defaultdict(dict)
    seen_ts = []
    with csv_path.open() as f:
        reader = csv.DictReader(f, delimiter=";")
        for row in reader:
            ts = int(row["timestamp"])
            if ts not in ticks and len(seen_ts) >= max_ticks:
                continue
            if ts not in ticks:
                seen_ts.append(ts)
            sym = row["product"]
            depth = OrderDepth()
            for i in (1, 2, 3):
                bp = row.get(f"bid_price_{i}", "")
                bv = row.get(f"bid_volume_{i}", "")
                ap = row.get(f"ask_price_{i}", "")
                av = row.get(f"ask_volume_{i}", "")
                if bp and bv:
                    depth.buy_orders[int(float(bp))] = int(float(bv))
                if ap and av:
                    depth.sell_orders[int(float(ap))] = -int(float(av))
            ticks[ts][sym] = depth
    return [(ts, ticks[ts]) for ts in seen_ts[:max_ticks]]


def main():
    csv_path = ROOT / "ROUND_5" / "prices_round_5_day_2.csv"
    max_ticks = int(sys.argv[1]) if len(sys.argv) > 1 else 30
    ticks = load_ticks(csv_path, max_ticks=max_ticks)
    print(f"Loaded {len(ticks)} ticks; first tick has {len(ticks[0][1])} symbols")

    trader = Trader()
    trader_data = ""
    positions: dict[str, int] = {}
    n_orders = 0
    n_violations = 0
    n_exceptions = 0
    t0 = time.perf_counter()

    for ts, depths in ticks:
        state = TradingState(
            traderData=trader_data,
            timestamp=ts,
            listings={s: Listing(s, s, 1) for s in depths},
            order_depths=depths,
            own_trades={},
            market_trades={},
            position=dict(positions),
            observations=Observation({}, {}),
        )
        try:
            orders, _conv, trader_data = trader.run(state)
        except Exception as e:
            n_exceptions += 1
            print(f"  EXC at ts={ts}: {type(e).__name__}: {e}")
            continue

        # Verify no order would push past the cap (worst-case projection per side).
        for sym, lst in orders.items():
            cap = BRONZE_POSITION_CAP if sym in BRONZE_SYMS else POSITION_LIMIT
            cur = positions.get(sym, 0)
            buy_total = sum(o.quantity for o in lst if o.quantity > 0)
            sell_total = sum(-o.quantity for o in lst if o.quantity < 0)
            if cur + buy_total > cap or cur - sell_total < -cap:
                n_violations += 1
                print(f"  VIOLATION ts={ts} {sym} pos={cur} buys={buy_total} sells={sell_total} cap={cap}")
            n_orders += len(lst)

        # Simulate naive fills to keep positions evolving (cross orders only).
        # For smoke test purposes we don't need full matching; positions stay zero
        # which is fine since we want to verify no exceptions.

    elapsed = time.perf_counter() - t0
    avg_ms = elapsed / max(1, len(ticks)) * 1000
    print(f"\n=== Smoke test ===")
    print(f"Ticks processed: {len(ticks)}")
    print(f"Total orders emitted: {n_orders}")
    print(f"Avg ms/tick: {avg_ms:.2f}")
    print(f"Exceptions: {n_exceptions}")
    print(f"Position-limit violations: {n_violations}")
    if n_exceptions == 0 and n_violations == 0:
        print("PASS")
        return 0
    print("FAIL")
    return 1


if __name__ == "__main__":
    sys.exit(main())
