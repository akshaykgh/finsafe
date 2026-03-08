from flask import Flask, request, jsonify
from flask_cors import CORS
import pandas as pd

from models.categorizer    import categorize_transactions, TRAINING_ROWS
from models.anomaly        import detect_anomalies
from models.merchant_risk  import compute_merchant_risk
from models.velocity_burst import detect_velocity_burst
from models.category_drift import detect_category_drift
from models.forecast       import forecast_spending
from utils.health_score    import compute_health_score

app = Flask(__name__)
CORS(app)


@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        "status":        "ok",
        "models":        "loaded",
        "training_rows": TRAINING_ROWS
    })


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.json.get('transactions', [])
        if not data:
            return jsonify({"error": "No transactions provided"}), 400

        df = pd.DataFrame(data)
        df.columns = df.columns.str.lower().str.strip()

        missing = [c for c in ['description', 'amount'] if c not in df.columns]
        if missing:
            return jsonify({"error": f"Missing columns: {missing}"}), 400

        df['amount'] = (
            df['amount'].astype(str)
            .str.replace(r'[$,]', '', regex=True)
        )
        df['amount'] = pd.to_numeric(df['amount'], errors='coerce').fillna(0).abs()

        if 'date' not in df.columns:
            df['date'] = pd.Timestamp.today()

        df = categorize_transactions(df)
        df = detect_anomalies(df)
        df = compute_merchant_risk(df)
        df = detect_velocity_burst(df)

        drift_alerts    = detect_category_drift(df)
        forecast        = forecast_spending(df)
        anomalies_count = int(df['anomaly'].sum())
        burst_count     = int(df['velocity_burst'].sum())
        health_score    = compute_health_score(df, anomalies_count, drift_alerts, burst_count)

        category_summary = (
            df.groupby('category')['amount']
            .sum().round(2).reset_index()
            .rename(columns={'amount': 'total'})
            .to_dict(orient='records')
        )

        df['date_parsed'] = pd.to_datetime(df['date'], errors='coerce')
        df['week_label']  = df['date_parsed'].dt.strftime('Week %W')
        weekly_spend = (
            df.groupby('week_label')['amount']
            .sum().round(2).reset_index()
            .rename(columns={'amount': 'total', 'week_label': 'week'})
            .to_dict(orient='records')
        )

        drop_cols = [
            'date_parsed', 'date_only', 'hour', 'hour_parsed',
            'week', 'week_label', 'month', 'rolling_3day'
        ]
        df = df.drop(columns=[c for c in drop_cols if c in df.columns])

        return jsonify({
            "transactions":       df.fillna('').to_dict(orient='records'),
            "category_summary":   category_summary,
            "weekly_spend":       weekly_spend,
            "forecast":           forecast,
            "drift_alerts":       drift_alerts,
            "health_score":       health_score,
            "anomalies_count":    anomalies_count,
            "burst_count":        burst_count,
            "total_transactions": len(df)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000, debug=True)
