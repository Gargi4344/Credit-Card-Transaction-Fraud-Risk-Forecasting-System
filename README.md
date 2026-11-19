# Credit Card Transaction Fraud & Risk Forecasting System

An end-to-end machine learning system built on the public **IEEE-CIS Fraud Detection** dataset (about 590,000 card transactions). It answers two questions:

1. **Is this transaction suspicious?** A gradient-boosting classifier scores each transaction and assigns a risk band.
2. **How much fraud should we expect over the next two weeks?** Daily fraud rates per customer segment are forecast 14 days ahead with a PyTorch LSTM, compared against an ARIMA baseline.

A risk or operations team can use the scores to triage individual transactions and the forecasts to plan review capacity.

## Pipeline

```
IEEE-CIS CSVs ──▶ PySpark feature pipeline ──▶ fraud_features (Parquet)
                                                    │
                       ┌────────────────────────────┴───────────────────────┐
                       ▼                                                    ▼
        Gradient-boosting classifier                       PySpark daily segment series
        (scikit-learn HistGradientBoosting)                (fraud rate per ProductCD × card network)
                       │                                                    │
                       │                                   ARIMA baseline + PyTorch LSTM
                       │                                   14-day forecasts per segment
                       ▼                                                    ▼
                         FastAPI service: /api/v1/score, /api/v1/forecasts
                                                    │
                                                    ▼
                                          Streamlit dashboard
```

## Components

| Folder | What it does |
|---|---|
| [spark/feature_pipeline.py](spark/feature_pipeline.py) | Reads the raw transaction and identity CSVs with PySpark, joins them, cleans card and email fields, and builds features: log-scaled amount, transaction day, per-card velocity counts over 1 hour and 24 hours, 24-hour spend per card, and a missing-field count |
| [spark/build_segment_series.py](spark/build_segment_series.py) | Aggregates the features into a daily fraud-rate time series for each segment (product code × card network) |
| [ml/classifier.py](ml/classifier.py) | Trains a `HistGradientBoostingClassifier` in a scikit-learn pipeline (imputation + categorical encoding) on a stratified 80/20 split, and reports ROC-AUC, PR-AUC and a classification report |
| [forecasting/](forecasting/) | ARIMA(1,1,1) baseline with confidence intervals (`statsmodels`), and a 2-layer PyTorch LSTM that forecasts 14 days ahead from a 21-day lookback window |
| [api/main.py](api/main.py) | FastAPI service that loads the trained model, scores transactions (probability + `low` / `medium` / `high` / `critical` band) and serves forecasts |
| [dashboard/app.py](dashboard/app.py) | Streamlit dashboard showing forecast trends by segment |
| [db/init.sql](db/init.sql) | PostgreSQL tables and indexes for transaction scores and segment forecasts |
| [tests/](tests/) | pytest tests for the classifier and the LSTM forecaster, run on every push by GitHub Actions |

## Tech stack

Python 3.12 · PySpark · pandas · scikit-learn · PyTorch · statsmodels · FastAPI · PostgreSQL · Streamlit · Docker Compose · GitHub Actions

## Getting started

### 1. Get the data

The dataset is not included in this repository. Download it from the [IEEE-CIS Fraud Detection competition on Kaggle](https://www.kaggle.com/c/ieee-fraud-detection/data) and place these files in `data/raw/`:

```
data/raw/train_transaction.csv
data/raw/train_identity.csv
```

### 2. Install dependencies

Requires Python 3.12. The Spark steps also need Java 17 and `pyspark`.

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pip install pyspark
```

### 3. Run the pipeline

```bash
make features   # PySpark feature engineering   -> data/processed/fraud_features
make segments   # daily fraud rate per segment  -> data/processed/segment_daily
make train      # train classifier, print metrics -> artifacts/fraud_classifier.joblib
make forecast   # ARIMA + LSTM 14-day forecasts -> artifacts/forecasts/
```

On Windows without `make`, run the commands listed in the [Makefile](Makefile) directly.

### 4. Start the API and dashboard

```bash
docker compose up --build
```

| Service | URL |
|---|---|
| API docs (Swagger) | http://localhost:8010/docs |
| Health check | http://localhost:8010/health |
| Dashboard | http://localhost:8510 |
| PostgreSQL | `localhost:5434` |

Example scoring request:

```bash
curl -X POST http://localhost:8010/api/v1/score \
  -H "Content-Type: application/json" \
  -d '{"TransactionAmt": 249.99, "TransactionDT": 86400, "ProductCD": "W", "card4": "visa", "card6": "debit", "segment_txn_1h": 4, "segment_txn_24h": 9, "segment_amt_24h": 1200}'
```

### 5. Run tests

```bash
make test
```

## Evaluation

The classifier is evaluated with ROC-AUC, PR-AUC, precision and recall. PR-AUC matters most here because only about 3.5% of transactions are fraud. Forecasts are compared with the ARIMA baseline using MAE and RMSE. Run `make train` and `make forecast` on the full dataset to reproduce the results.
