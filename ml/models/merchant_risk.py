import pandas as pd

_HIGH_RISK_KEYWORDS = [
    'unknown wire', 'offshore', 'international wire', 'casino',
    'gambling', 'betting', 'crypto exchange', 'bitcoin atm',
    'western union', 'moneygram', 'money gram'
]

_MEDIUM_RISK_KEYWORDS = [
    'atm withdrawal', 'cash advance', 'zelle', 'venmo', 'paypal', 'cashapp',
    'wire transfer', 'cash app'
]

_LATE_NIGHT_START = 0   # midnight
_LATE_NIGHT_END   = 5   # 5 AM exclusive
_LATE_NIGHT_FLOOR = 200 # flag late-night transactions above this amount


def compute_merchant_risk(df):
    df = df.copy()
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df['_tx_hour']    = df['date_parsed'].dt.hour.fillna(12).astype(int)

    merchant_counts = df['description'].str.lower().value_counts()
    avg_count       = merchant_counts.mean()

    fraud_signals  = []
    risk_levels    = []

    for _, row in df.iterrows():
        desc   = str(row['description']).lower()
        amount = float(row['amount'])
        hour   = int(row['_tx_hour'])
        signals = []

        # Keyword-based fraud signals
        for kw in _HIGH_RISK_KEYWORDS:
            if kw in desc:
                signals.append(f"High-risk keyword: '{kw}'")
                break

        # Late-night + high-amount flag
        if _LATE_NIGHT_START <= hour < _LATE_NIGHT_END and amount > _LATE_NIGHT_FLOOR:
            signals.append(f"Late-night charge ${amount:.0f} at {hour:02d}:00")

        # Risk tier
        if any(kw in desc for kw in _HIGH_RISK_KEYWORDS):
            tier = 'High Risk'
        elif any(kw in desc for kw in _MEDIUM_RISK_KEYWORDS):
            tier = 'Medium Risk'
        elif merchant_counts.get(desc, 0) >= avg_count:
            tier = 'Familiar'
        elif merchant_counts.get(desc, 0) > 1:
            tier = 'Infrequent'
        else:
            tier = 'New'

        fraud_signals.append(' | '.join(signals))
        risk_levels.append(tier)

    df['fraud_signal'] = fraud_signals
    df['merchant_risk'] = risk_levels
    df = df.drop(columns=['_tx_hour'])
    return df
