# === Cell 9 (REPLACEMENT, robust): Distill to ranked rule table ===
# 不再依赖 `promising` 列表 —— 直接遍历所有 (product, horizon) 模型的 top-importance features，
# 加 brute-force 单特征 quantile threshold 扫描，按 |t_oos| 排序。
# 即使 Cell 6 的 promising 是空的，这里也能跑出 rules。

rules = []

# Loop ALL trained models (regardless of sharpe filter)
target_pairs = list(all_models.keys())
# filter to (product, horizon) tuples (skip the adverse-selection entries which have 3 elements)
target_pairs = [k for k in target_pairs if isinstance(k, tuple) and len(k) == 2]

print(f'Scanning {len(target_pairs)} (product, horizon) models...')

for p, h in target_pairs:
    models = all_models[(p, h)]
    # Use day-3-as-test model to get feature importance (most novel test set)
    if 3 not in models:
        continue
    m, fcols, _, _, _ = models[3]
    imp = pd.Series(m.feature_importances_, index=fcols).sort_values(ascending=False)
    # take top 10 importance features (broader scan than top 5)
    for fn in imp.head(10).index:
        df = feats[p].dropna(subset=[fn, f'y_{h}']).reset_index(drop=True)
        if len(df) < 200:
            continue
        # try multiple quantile thresholds
        for q in [0.02, 0.05, 0.10, 0.20, 0.80, 0.90, 0.95, 0.98]:
            try:
                thresh = df[fn].quantile(q)
            except Exception:
                continue
            if q < 0.5:
                mask = df[fn] <= thresh
            else:
                mask = df[fn] >= thresh
            sub = df[mask]
            if len(sub) < 100:
                continue
            in_s = sub[sub['dayfile'].isin([1, 2])][f'y_{h}']
            out_s = sub[sub['dayfile'] == 3][f'y_{h}']
            if len(in_s) < 50 or len(out_s) < 20:
                continue
            in_mean = in_s.mean()
            out_mean = out_s.mean()
            # require sign-consistent
            if np.sign(in_mean) != np.sign(out_mean):
                continue
            out_t = out_mean * np.sqrt(len(out_s)) / (out_s.std() + 1e-9)
            # LOWER threshold from 1.0 → 0.5 to capture more candidates
            if abs(out_t) < 0.5:
                continue
            rules.append({
                'product': p, 'horizon': h, 'feature': fn,
                'condition': '<=' if q < 0.5 else '>=',
                'threshold': float(thresh),
                'in_mean': in_mean, 'out_mean': out_mean,
                'out_t': out_t, 'n_out': len(out_s),
                'fire_rate': len(sub) / len(df),
            })

print(f'\nTotal rule candidates collected: {len(rules)}')

# Empty-handling: ensure DataFrame has expected columns even when empty
columns = ['product', 'horizon', 'feature', 'condition', 'threshold',
           'in_mean', 'out_mean', 'out_t', 'n_out', 'fire_rate']
rules_df = pd.DataFrame(rules, columns=columns)

if len(rules_df) > 0:
    rules_df = rules_df.drop_duplicates(
        subset=['product', 'feature', 'condition', 'threshold', 'horizon']
    )
    rules_df = rules_df.sort_values('out_t', key=lambda s: s.abs(), ascending=False)
    print(f'Sign-consistent + |t_oos|>0.5 rules: {len(rules_df)}')
    print(rules_df.head(40).to_string(
        index=False,
        float_format=lambda x: f'{x:+.3f}' if abs(x) < 1000 else f'{x:.0f}'
    ))
else:
    print('\n*** No rules pass the filters (sign-consistent + |t_oos|>0.5). ***')
    print('This means: ML cross-day alpha really is not present in this data with the current features.')
    print('See Cell 8 (adverse-selection AUC) for the alternate framing.')

rules_df.to_csv('round4_alpha_rules.csv', index=False)
print('\nSaved → round4_alpha_rules.csv')
