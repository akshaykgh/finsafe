def compute_health_score(df, anomalies_count, drift_alerts, burst_count):
    score = 100
    score -= min(anomalies_count * 5, 30)   # cap at -30
    score -= min(len(drift_alerts) * 8, 24) # cap at -24
    score -= min(burst_count * 2, 20)        # cap at -20, reduce multiplier
    total = df['amount'].sum()
    if total > 0:
        max_share = df.groupby('category')['amount'].sum().max() / total
        if max_share > 0.4:
            score -= 10
    if 'Income' in df['category'].values:
        score += 10
    return max(0, min(100, round(score)))
