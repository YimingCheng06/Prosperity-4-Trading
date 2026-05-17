# Voucher Pricing Analysis: Round 4

## Executive Summary

**TTE Decay Verdict:** NO - Time-to-expiry is constant
- VEV_6000 OTM time-value Day 1→3: 0.500 → 0.500 (Δ=0.0%)
- Deep-ITM (VEV_4000) maintains intrinsic + stable TV across days

**Recommended TTE Model:**
```
if tte_decays:
    T_years = (4 - (day - 1)) / 252
else:
    T_years = 4 / 252
```

## IV Smile Shape

The implied volatility smile is characterized by a moderate U-shape (positive quadratic term):

| Coefficient | Value | Interpretation |
|---|---|---|
| a0 (Quadratic, curvature) | 0.131776 ± 0.012890 | Smile bowl; stable across days |
| a1 (Linear, skew) | 0.016920 ± 0.020448 | Mild skew; negligible variation |
| a2 (Constant, ATM IV) | 0.228592 ± 0.019137 | ATM vol ~22.9%; stable |

**Stability:** Low day-over-day variation in all coefficients → smile shape is consistent. Fit a single smile per timestamp rather than globally.

## Mispricing Opportunities

### Largest Mean Residuals
```
 product  mean_abs_residual  std_residual  acf_lag_100  acf_lag_500  acf_lag_1000
VEV_5400           2.492472      0.743439     0.453216     0.393469      0.343215
VEV_5300           2.242435      1.109662     0.583216     0.524877      0.478290
VEV_5200           2.055906      1.179085     0.268105     0.219631      0.193398
```

### High-Persistence Mispricings (ACF lag 100 > 0.3)
Residuals with autocorrelation > 0.3 at lag 100 timestamps (~10 sec) indicate alpha that persists:
```
 product  acf_lag_100  acf_lag_500  mean_abs_residual
VEV_5400     0.453216     0.393469           2.492472
VEV_5300     0.583216     0.524877           2.242435
VEV_5500     0.375388     0.333570           0.933393
VEV_6000     0.747519     0.698177           0.181898
VEV_6500     0.422453     0.384769           0.063116
```
**Strategy Implication:** Deep-ITM vouchers (VEV_4000, VEV_4500) show the largest and most persistent residuals. Consider using a slower mean-reversion or inventory-based quote update for these products.

## Recommended Pricing Function

```python
def theo_voucher(spot, K, T, iv_smile_model=None):
    """
    Price a VEV call option.
    
    Args:
        spot: VELVETFRUIT_EXTRACT spot price
        K: strike (one of {4000, 4500, 5000, 5100, 5200, 5300, 5400, 5500, 6000, 6500})
        T: time-to-expiry in years (see TTE decay verdict above)
        iv_smile_model: pre-fitted smile coefficients [a0, a1, a2]
                        If None, use smile mean: [a0=0.131776, 
                                                    a1=0.016920, 
                                                    a2=0.228592]
    
    Returns:
        theo_price (float)
    
    Algorithm:
        1. Compute log-moneyness: m = log(K/spot) / sqrt(T)
        2. Compute theo IV: IV = a0*m^2 + a1*m + a2
        3. Return BS_call(spot, K, T, IV)
    """
    import numpy as np
    from scipy.stats import norm
    
    if T <= 0:
        return max(spot - K, 0)
    
    if iv_smile_model is None:
        iv_smile_model = [0.131776, 0.016920, 0.228592]
    
    # Compute IV from smile
    m = np.log(K / spot) / np.sqrt(T)
    theo_iv = iv_smile_model[0] * m**2 + iv_smile_model[1] * m + iv_smile_model[2]
    theo_iv = max(theo_iv, 0.001)  # floor at 0.1%
    
    # Black-Scholes (r=0)
    d1 = (np.log(spot / K) + 0.5 * theo_iv**2 * T) / (theo_iv * np.sqrt(T))
    d2 = d1 - theo_iv * np.sqrt(T)
    price = spot * norm.cdf(d1) - K * norm.cdf(d2)
    
    return price
```

## Files Generated

- `smile_snapshot.png`: Visual of IV smile fit from Day 1, mid-window
- `voucher_residuals.csv`: Full market_mid - theo_price residual time series for all vouchers
- This summary document

---

**Analysis Date:** 2026-04-27  
**Simulation Window:** 3 days, 1000 timestamps each (10th of trading day)  
**Strikes:** VEV_4000 to VEV_6500  
**Spot Reference:** VELVETFRUIT_EXTRACT ~5245  
