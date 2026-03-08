import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

_SKIP_AMOUNT_CHECK = {'Income'}


def _build_features(df):
    df = df.copy()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['hour']        = df['date_parsed'].dt.hour.fillna(12).astype(int)
    df['date_only']   = df['date_parsed'].dt.date
    df['week']        = df['date_parsed'].dt.isocalendar().week.astype(int)

    df['amount_zscore']      = (df['amount'] - df['amount'].mean()) / (df['amount'].std() + 1e-9)
    cat_means                = df.groupby('category')['amount'].transform('mean')
    cat_stds                 = df.groupby('category')['amount'].transform('std').fillna(1)
    df['category_deviation'] = (df['amount'] - cat_means) / (cat_stds + 1e-9)

    user_hour_mean       = df['hour'].mean()
    user_hour_std        = df['hour'].std() if df['hour'].std() > 0 else 1
    df['hour_deviation'] = (df['hour'] - user_hour_mean) / user_hour_std

    daily_counts             = df.groupby('date_only')['amount'].transform('count')
    avg_daily                = daily_counts.mean()
    std_daily                = daily_counts.std() if daily_counts.std() > 0 else 1
    df['velocity_deviation'] = (daily_counts - avg_daily) / std_daily

    weekly_avg          = df.groupby('week')['amount'].transform('mean')
    df['vs_weekly_avg'] = df['amount'] / (weekly_avg + 1e-9)

    return df[[
        'amount_zscore', 'category_deviation',
        'hour_deviation', 'velocity_deviation', 'vs_weekly_avg'
    ]].fillna(0)


def _build_reason(row, df, hour_mean, hour_std):
    reason_parts = []

    if row.get('category') not in _SKIP_AMOUNT_CHECK:
        cat_avg = df[df['category'] == row['category']]['amount'].mean()
        if cat_avg > 0 and row['amount'] > cat_avg * 2:
            reason_parts.append(
                f"{round(row['amount'] / cat_avg, 1)}x your usual {row['category']} spend"
            )

    tx_hour = pd.to_datetime(row['date'], errors='coerce').hour
    if abs(tx_hour - hour_mean) / (hour_std + 1e-9) > 2:
        reason_parts.append(
            f"outside your normal hours (usually around {int(hour_mean)}:00)"
        )

    date_only = pd.to_datetime(row['date'], errors='coerce').date()
    day_count = len(df[df['date_parsed'].dt.date == date_only])
    avg_daily = df.groupby(df['date_parsed'].dt.date).size().mean()
    if day_count > avg_daily * 2:
        reason_parts.append(
            f"{day_count} transactions today vs your average {round(avg_daily, 1)}/day"
        )

    return " - ".join(reason_parts) if reason_parts else "Unusual pattern vs your personal history"


def detect_anomalies(df):
    if len(df) < 10:
        df['anomaly']        = False
        df['anomaly_reason'] = ''
        df['anomaly_score']  = 0.0
        return df

    df = df.copy()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['hour_parsed'] = df['date_parsed'].dt.hour.fillna(12).astype(int)

    hour_mean = df['hour_parsed'].mean()
    hour_std  = df['hour_parsed'].std() if df['hour_parsed'].std() > 0 else 1

    features    = _build_features(df)
    spend_mask  = (df['category'] != 'Income').values
    income_mask = ~spend_mask

    iso    = IsolationForest(contamination=0.05, random_state=42, n_estimators=100)
    iso.fit(features[spend_mask])
    preds  = iso.predict(features)
    scores = iso.score_samples(features)
    preds[income_mask] = 1

    df['anomaly']        = preds == -1
    df['anomaly_score']  = scores.round(3)
    df['anomaly_reason'] = [
        _build_reason(row, df, hour_mean, hour_std) if row['anomaly'] else ''
        for _, row in df.iterrows()
    ]
    return df
