import pandas as pd
from sklearn.linear_model import LinearRegression


def forecast_spending(df):
    try:
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'], errors='coerce')

        # Use a day offset from min date to avoid ISO week year-boundary bugs
        min_date       = df['date'].min()
        df['day_offset'] = (df['date'] - min_date).dt.days
        df['week_bucket'] = df['day_offset'] // 7

        weekly = df.groupby('week_bucket')['amount'].sum().reset_index()
        weekly.columns = ['week', 'total']

        if len(weekly) < 3:
            avg = weekly['total'].mean() if len(weekly) > 0 else 100
            return [{"week": f"Week +{i+1}", "forecast": round(avg, 2)} for i in range(4)]

        model = LinearRegression()
        model.fit(weekly['week'].values.reshape(-1, 1), weekly['total'].values)
        last_week   = int(weekly['week'].max())
        rolling_avg = weekly['total'].tail(4).mean()

        forecasts = []
        for i in range(1, 5):
            linear_pred = model.predict([[last_week + i]])[0]
            # Blend recent rolling average with linear trend to dampen runaway extrapolation
            blended = 0.6 * rolling_avg + 0.4 * linear_pred
            forecasts.append({
                "week":     f"Week +{i}",
                "forecast": round(max(blended, 0), 2)
            })
        return forecasts

    except Exception:
        return [{"week": f"Week +{i}", "forecast": 0.0} for i in range(1, 5)]
