features:
	python spark/feature_pipeline.py
segments:
	python spark/build_segment_series.py
train:
	python ml/classifier.py
forecast:
	PYTHONPATH=forecasting python forecasting/run_forecast.py
test:
	PYTHONPATH=. pytest -q tests
api:
	uvicorn api.main:app --reload --port 8010
