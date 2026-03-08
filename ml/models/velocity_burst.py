import pandas as pd


def detect_velocity_burst(df):
    df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
    df = df.sort_values('date_parsed').copy()

    # Exclude income from burst detection — paycheck days should not trigger alerts
    spend_only = df[df['category'] != 'Income']['amount']
    spend_series = df['amount'].where(df['category'] != 'Income', 0)

    df['rolling_3day'] = (
        spend_series
        .set_axis(df['date_parsed'])
        .rolling('3D')
        .sum()
        .values
    )

    avg_3day = df['rolling_3day'].mean()
    std_3day = df['rolling_3day'].std() if df['rolling_3day'].std() > 0 else 1

    df['velocity_burst']     = df['rolling_3day'] > (avg_3day + 2.5 * std_3day)
    df['velocity_burst_msg'] = df.apply(
        lambda r: (
            f"Spent ${r['rolling_3day']:.0f} in 3 days - "
            f"{round(r['rolling_3day'] / avg_3day, 1)}x your normal 3-day window"
        ) if r['velocity_burst'] else '',
        axis=1
    )
    return df

