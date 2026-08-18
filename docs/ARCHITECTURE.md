# Architecture

PySpark handles distributed feature engineering. The classification path uses a tree model for point-in-time transaction risk. The forecasting path aggregates fraud rate by segment and compares a classical ARIMA baseline with a PyTorch LSTM. FastAPI exposes both scoring and forecasts; Streamlit provides operational visualization.

Evaluate classification with ROC-AUC, PR-AUC, recall, precision and threshold-specific confusion matrices. Evaluate forecasts with MAE/RMSE by horizon and segment. Do not publish metrics until the real dataset has been processed.
