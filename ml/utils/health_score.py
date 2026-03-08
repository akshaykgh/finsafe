import pandas as pd


def compute_health_score(df, anomalies_count, drift_alerts, burst_count):
    score = 100

    score -= min(anomalies_count * 5, 30)   # cap at -30
    score -= min(len(drift_alerts) * 8, 24) # cap at -24
    score -= min(burst_count * 2, 20)       # cap at -20

    spend_df      = df[df['category'] != 'Income']
    total_spend   = spend_df['amount'].sum()
    total_income  = df[df['category'] == 'Income']['amount'].sum()

    # Penalize if one spend category dominates
    if total_spend > 0:
        max_share = spend_df.groupby('category')['amount'].sum().max() / total_spend
        if max_share > 0.4:
            score -= 10

    # Income bonus only when income meaningfully covers spending
    if total_income > 0 and total_spend > 0:
        ratio = total_income / total_spend
        if ratio >= 1.2:
            score += 10
        elif ratio >= 1.0:
            score += 5

    return max(0, min(100, round(score)))
