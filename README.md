# FinSafe ML Service

Intelligent transaction analysis engine for FinSafe. Accepts raw bank transactions and returns categorization, anomaly detection, fraud signals, spending forecasts, and a financial health score.

---

## Tech Stack

- **Python 3.10+**
- **FastAPI** — REST API
- **scikit-learn** — TF-IDF categorizer + IsolationForest anomaly detection
- **pandas / numpy** — data processing
- **Hugging Face datasets** — primary training data source

---

## Project Structure

```
ml/
├── app.py                  # FastAPI entry point, /predict endpoint
├── models/
│   ├── categorizer.py      # Transaction categorizer (TF-IDF + Logistic Regression)
│   ├── anomaly.py          # Anomaly detection (IsolationForest)
│   ├── merchant_risk.py    # Fraud signal + merchant risk tier
│   ├── velocity_burst.py   # 3-day spending burst detection
│   ├── category_drift.py   # Month-over-month category drift alerts
│   └── forecast.py         # 4-week spend forecast (Linear Regression)
└── utils/
    ├── data_loader.py      # Training data — Hugging Face + fallback keywords
    └── health_score.py     # 0–100 financial health score
```

---

## Setup

```bash
cd ml
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

Service runs on `http://localhost:8000` by default.

---

## Endpoints

### `GET /health`
Returns service status and number of training rows loaded.

**Response:**
```json
{
  "status": "ok",
  "training_rows": 21400
}
```

---

### `POST /predict`
Accepts a list of raw transactions and returns full analysis.

**Request body** (sent by Node.js backend):
```json
{
  "transactions": [
    {
      "date": "2025-12-01T08:15:00.000Z",
      "description": "Walmart groceries",
      "amount": -93.10
    }
  ]
}
```

> `amount` can be negative (debit) or positive (credit) — the service takes `abs()` internally.

**Response:**
```json
{
  "health_score": 66,
  "total_transactions": 66,
  "anomalies_count": 4,
  "burst_count": 2,
  "category_summary": [
    {
      "category": "Food & Dining",
      "total": 457.23,
      "avg_confidence": 96.9
    }
  ],
  "drift_alerts": [
    {
      "category": "Financial Services",
      "current_pct": 25.4,
      "usual_pct": 7.4,
      "delta_pct": 17.9,
      "message": "Financial Services jumped to 25% of your budget this month vs your usual 7%"
    }
  ],
  "forecast": [
    { "week": "Week +1", "forecast": 3132.06 },
    { "week": "Week +2", "forecast": 2592.88 },
    { "week": "Week +3", "forecast": 2053.70 },
    { "week": "Week +4", "forecast": 1514.52 }
  ],
  "transactions": [
    {
      "date": "2025-12-01 08:15",
      "description": "Walmart groceries",
      "amount": 93.10,
      "category": "Food & Dining",
      "category_confidence": 94.3,
      "anomaly": false,
      "anomaly_reason": "",
      "anomaly_score": -0.43,
      "fraud_signal": "",
      "merchant_risk": "Familiar",
      "velocity_burst": false,
      "velocity_burst_msg": ""
    }
  ]
}
```

---

## Categories

| Category | Examples |
|---|---|
| Food & Dining | Subway, Grubhub, Doordash, Walmart groceries |
| Transportation | Uber, Lyft, Shell gas, Metro card |
| Shopping & Retail | Amazon, Target, Best Buy, Nike |
| Utilities & Services | Verizon, Comcast, Dropbox, Microsoft 365 |
| Rent & Mortgage | Rent payment, Monthly rent, Mortgage |
| Entertainment & Recreation | Netflix, Spotify, Planet Fitness, AMC |
| Healthcare & Medical | CVS, Dental clinic, Urgent care |
| Financial Services | Venmo, Zelle, PayPal, Wire transfers |
| Income | Paycheck, Cashback reward, Refund credit |
| Gambling | DraftKings, Offshore casino, Betting |
| Charity & Donations | Red Cross, GoFundMe, Patreon |
| Government & Legal | IRS payment, DMV fee, Parking ticket |

---

## Anomaly Flags

Each transaction gets an `anomaly_score` from IsolationForest (lower = more anomalous). A transaction is flagged when:

- Amount is **2x+ the category average**
- Transaction occurs **outside normal hours** (2+ std deviations from mean hour)
- **Daily transaction count** is 2x the daily average

`Income` and `Rent & Mortgage` categories are excluded from anomaly detection to prevent false positives on paychecks and rent.

---

## Fraud Signals

High-risk keywords that trigger `fraud_signal`:

- `unknown wire`, `offshore`, `international wire`
- `casino`, `gambling`, `betting`
- `crypto exchange`, `bitcoin atm`, `western union`

Late-night charges (midnight–5 AM) above $200 are also flagged regardless of keyword.

---

## Training Data

On startup, `data_loader.py` loads training data in this order:

1. **Hugging Face** — `mitulshah/transaction-categorization` (20,000 rows)
2. **Fallback keyword templates** — always appended on top of HuggingFace data

If HuggingFace is unavailable, only fallback templates are used (~1,200 rows).

---

## Node.js Integration

The Node.js backend (port 8080) calls this service via:

```
POST ${ML_SERVICE_URL}/predict
```

Set `ML_SERVICE_URL` in the Node.js `.env`:

```env
ML_SERVICE_URL=http://localhost:8000
```

The Node.js backend maps the response fields `categorizedTransactions`, `flaggedTransactions`, and `forecast` from the ML response. The ML service returns these under `transactions`, `transactions` (filtered where `anomaly: true`), and `forecast` respectively.

---

## Notes

- Model is trained fresh on every cold start (~5–10 seconds).
- All data is processed in-memory — no database required.
- To productionize: serialize the trained pipeline with `joblib` and load from disk to eliminate cold start.
