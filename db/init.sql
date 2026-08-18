CREATE TABLE IF NOT EXISTS transaction_scores (transaction_id VARCHAR(128) PRIMARY KEY, fraud_probability DOUBLE PRECISION NOT NULL, predicted_label INTEGER NOT NULL, model_version VARCHAR(128) NOT NULL, scored_at TIMESTAMPTZ DEFAULT NOW());
CREATE TABLE IF NOT EXISTS segment_forecasts (id BIGSERIAL PRIMARY KEY, segment_key VARCHAR(200) NOT NULL, forecast_date DATE NOT NULL, model_name VARCHAR(64) NOT NULL, predicted_fraud_rate DOUBLE PRECISION NOT NULL, lower_bound DOUBLE PRECISION, upper_bound DOUBLE PRECISION, created_at TIMESTAMPTZ DEFAULT NOW());
CREATE INDEX IF NOT EXISTS idx_score_prob ON transaction_scores(fraud_probability DESC);
CREATE INDEX IF NOT EXISTS idx_forecast_segment_date ON segment_forecasts(segment_key, forecast_date);
