from pathlib import Path
import joblib,pandas as pd,math
from fastapi import FastAPI,HTTPException,Query
from pydantic import BaseModel,Field
app=FastAPI(title="Fraud Risk Forecasting API",version="1.0")
MODEL=Path("artifacts/fraud_classifier.joblib");FORECAST=Path("artifacts/forecasts/forecast_results.parquet");model=None
class Transaction(BaseModel):
 TransactionAmt:float=Field(gt=0);TransactionDT:float=Field(ge=0);ProductCD:str="W";card4:str="unknown";card6:str="unknown";P_emaildomain:str="unknown";amount_log1p:float|None=None;segment_txn_24h:float=0;segment_txn_1h:float=0;segment_amt_24h:float=0;missing_field_count:float=0
@app.on_event("startup")
def load():
 global model
 if MODEL.exists():model=joblib.load(MODEL)
@app.get("/health")
def health():return {"status":"ok","model_loaded":model is not None}
@app.post("/api/v1/score")
def score(t:Transaction):
 if model is None:raise HTTPException(503,"Train the classifier first")
 d=t.model_dump();d["amount_log1p"]=d["amount_log1p"] if d["amount_log1p"] is not None else math.log1p(d["TransactionAmt"]);prob=float(model.predict_proba(pd.DataFrame([d]))[:,1][0])
 return {"fraud_probability":prob,"risk_band":"critical" if prob>=.8 else "high" if prob>=.5 else "medium" if prob>=.2 else "low"}
@app.get("/api/v1/forecasts")
def forecasts(segment:str|None=None,limit:int=Query(100,ge=1,le=5000)):
 if not FORECAST.exists():raise HTTPException(404,"Forecasts not generated")
 df=pd.read_parquet(FORECAST);df=df[df.segment_key.astype(str).str.contains(segment,case=False,na=False)] if segment else df
 return df.head(limit).to_dict(orient="records")
