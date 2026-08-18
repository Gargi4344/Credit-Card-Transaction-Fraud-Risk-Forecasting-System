import numpy as np,pandas as pd
from statsmodels.tsa.arima.model import ARIMA
def persistence(values,horizon): return np.repeat(float(values[-1]),horizon)
def arima(values,horizon,order=(1,1,1)):
 s=pd.Series(values).astype(float).replace([np.inf,-np.inf],np.nan).dropna()
 if len(s)<12: return persistence(s.values,horizon),None,None
 fit=ARIMA(s,order=order).fit();res=fit.get_forecast(horizon);ci=res.conf_int()
 return np.asarray(res.predicted_mean),np.asarray(ci.iloc[:,0]),np.asarray(ci.iloc[:,1])
