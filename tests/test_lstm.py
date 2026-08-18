import numpy as np
from forecasting.lstm import train,forecast
def test_forecast():
 v=np.clip(np.linspace(.01,.1,80),0,1);m,_=train(v,epochs=2);o=forecast(m,v,14);assert len(o)==14;assert np.all(o>=0);assert np.all(o<=1)
