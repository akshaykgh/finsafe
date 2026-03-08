import pandas as pd

_MIN_MONTHLY_TX = 5  # ignore sparse months to avoid skewing the average


def detect_category_drift(df):
    try:
        df = df.copy()
        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        df['month']       = df['date_parsed'].dt.to_period('M')

        if df['month'].nunique() < 2:
            return []

        # Filter out months that have too few transactions (e.g. data gaps like Jan 2026)
        month_tx_counts = df.groupby('month').size()
        valid_months    = month_tx_counts[month_tx_counts >= _MIN_MONTHLY_TX].index
        df_valid        = df[df['month'].isin(valid_months)]

        if df_valid['month'].nunique() < 2:
            return []

        monthly_totals = df_valid.groupby('month')['amount'].sum()
        monthly_cat    = df_valid.groupby(['month', 'category'])['amount'].sum()
        monthly_share  = (monthly_cat / monthly_totals).reset_index()
        monthly_share.columns = ['month', 'category', 'share']

        avg_share    = monthly_share.groupby('category')['share'].mean()
        latest_month = df_valid['month'].max()
        latest       = monthly_share[monthly_share['month'] == latest_month]

        if latest.empty:
            return []

        latest_dict = dict(zip(latest['category'], latest['share']))

        alerts = []
        for cat, current in latest_dict.items():
            if cat in avg_share.index:
                delta = current - avg_share[cat]
                if delta > 0.10:
                    alerts.append({
                        "category":    cat,
                        "current_pct": round(current * 100, 1),
                        "usual_pct":   round(avg_share[cat] * 100, 1),
                        "delta_pct":   round(delta * 100, 1),
                        "message": (
                            f"{cat} jumped to {round(current * 100)}% of your budget "
                            f"this month vs your usual {round(avg_share[cat] * 100)}%"
                        )
                    })
        return alerts

    except Exception:
        return []
