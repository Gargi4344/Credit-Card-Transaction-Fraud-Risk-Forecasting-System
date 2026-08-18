from pathlib import Path
import numpy as np,pandas as pd
from baselines import arima
from lstm import train,forecast,save
def main():
 df=pd.read_parquet("data/processed/segment_daily");segments=[c for c in ["ProductCD","card4"] if c in df.columns];rows=[];Path("artifacts/forecasts").mkdir(parents=True,exist_ok=True)
 for key,g in df.groupby(segments,dropna=False):
  g=g.sort_values("transaction_day");v=g.fraud_rate.fillna(0).clip(0,1).to_numpy()
  if len(v)<21:continue
  a,lo,hi=arima(v,14);m,loss=train(v);l=forecast(m,v,14);save(m,f"artifacts/forecasts/lstm_{abs(hash(str(key)))}.pt")
  sk="|".join(str(x) for x in (key if isinstance(key,tuple) else (key,)))
  for i in range(14):rows.append({"segment_key":sk,"forecast_date":str(pd.Timestamp.today().normalize()+pd.Timedelta(days=i+1)).split(" ")[0],"arima":float(np.clip(a[i],0,1)),"lstm":float(l[i]),"lstm_train_mse":loss,"lower":None if lo is None else float(np.clip(lo[i],0,1)),"upper":None if hi is None else float(np.clip(hi[i],0,1))})
 out=pd.DataFrame(rows);out.to_parquet("artifacts/forecasts/forecast_results.parquet",index=False);out.to_csv("artifacts/forecasts/forecast_results.csv",index=False);print("forecast rows",len(out))
if __name__=="__main__":main()
