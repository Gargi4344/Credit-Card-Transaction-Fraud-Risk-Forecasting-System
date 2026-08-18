# Credit Card Transaction Fraud & Risk Forecasting System

Real-data ML engineering project designed for the IEEE-CIS Fraud Detection dataset (~590K transactions). The dataset is intentionally not redistributed. Put `train_transaction.csv` and `train_identity.csv` under `data/raw/`.

Pipeline: public CSV -> PySpark feature engineering -> fraud classifier -> daily segment series -> ARIMA baseline + PyTorch LSTM forecaster -> FastAPI -> PostgreSQL -> Streamlit dashboard.

The classifier answers whether an individual transaction is suspicious. The forecasting layer predicts fraud-rate trends by segment so an operations/risk team can prioritize future review capacity.

No performance numbers are fabricated. Run the supplied evaluation scripts on the downloaded public data before putting metrics on a resume.
