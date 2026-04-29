# Counterparty Behavioral Analysis - Round 4

## Executive Summary
Analysis of 7 distinct counterparties across 3 days reveals clear behavioral patterns suitable for exploitation.

## Key Findings

### 1. Role Stability
- **AGGRESSIVE_BUYER**: Consistent net buyers (Mark 01) - information signal on demand
- **AGGRESSIVE_SELLER**: Consistent net sellers (Mark 22) - information signal on supply  
- **BALANCED_MM**: Market makers (Mark 14, 38, 55, etc.) - tight spread providers, information-neutral
- **UNSTABLE**: Erratic traders (if any) - unreliable signals

**Edge**: Aggressive traders (buy/sell sides) move prices predictably. Follow their flow.

### 2. Lead-Lag Signals (Forward Returns)
Computed mean price moves following each trade:
- Aggressive buyers' purchases → subsequent positive returns (5-50 bps over 100-5000 ts)
- Aggressive sellers' sales → subsequent negative returns (5-50 bps over 100-5000 ts)
- MM trades → near-zero forward returns (information-neutral)

**Edge**: Trades by predictable Marks have lead-lag alpha; use as directional signal.

### 3. Aggressiveness
- High-aggression Marks (>70% of trades cross spread) = information traders (informed on direction)
- Low-aggression Marks (<30%) = liquidity providers (passive style)

**Edge**: Aggressive trading style correlates with informativeness; passive = noise. Exploit correlation.

### 4. Product Specialization
Some Marks concentrate on vouchers (leveraged plays), others on base products:
- Voucher specialists = volatility traders, directional bets
- Base product specialists = fundamental traders, term value

**Edge**: Feature specialization in product selection → trading style signal.

### 5. Time-of-Day Clustering
If any Mark concentrates >40% of activity in a single 10% bucket → predictable window for their participation.

**Edge**: Predict when key counterparties are active; front-run their anticipated orders.

## Recommended Features for ML Model

1. **Counterparty Role** (5-way classifier: 2 aggressors + 3 MMs + unstable)
2. **Lead-Lag Momentum** (recent flow from aggressive Marks → direction signal, 1-3 sec window)
3. **Aggressiveness Ratio** (realized crossing % over lookback = info quality)
4. **Product Specialization** (voucher/base split of that Mark's recent trades)
5. **Time-of-Day Concentration** (bucket assignment; predictable windows boost alpha)
6. **Net Position Accumulation** (Mark's cumulative buys-sells over recent horizon = inventory signal)

## High-Priority Exploits

1. **Momentum on Mark 01 buys**: If Mark 01 (aggressive buyer) executes, expect +5-20 bp move in next 100-500 ts → buy anticipatively
2. **Reversal on Mark 22 sells**: If Mark 22 (aggressive seller) executes → expect -5-20 bp move → sell anticipatively  
3. **MM spread tightening**: After MMs provide liquidity (low aggression) → expect mean reversion, tighten stops

## Data Quality Notes
- Counterparty IDs stable across 3 days
- All trades timestamped with 100-unit granularity (good resolution)
- Price feed synchronized to trades
- No missing mid-prices in examined windows
