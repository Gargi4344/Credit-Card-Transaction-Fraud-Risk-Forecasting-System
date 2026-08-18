import os,requests,pandas as pd,streamlit as st
API=os.getenv("API_BASE_URL","http://localhost:8010")
st.set_page_config(page_title="Fraud Risk Forecasting",layout="wide");st.title("Credit Card Fraud Risk Forecasting")
try:
 data=requests.get(f"{API}/api/v1/forecasts?limit=5000",timeout=10).json();df=pd.DataFrame(data)
 if df.empty:st.warning("Generate forecasts first")
 else:
  a,b,c=st.columns(3);a.metric("Forecast rows",len(df));b.metric("Segments",df.segment_key.nunique());c.metric("Peak LSTM rate",f"{df.lstm.max():.3%}")
  st.line_chart(df.pivot_table(index="forecast_date",columns="segment_key",values="lstm",aggfunc="mean"));st.dataframe(df,use_container_width=True)
except Exception as e:st.error(str(e))
