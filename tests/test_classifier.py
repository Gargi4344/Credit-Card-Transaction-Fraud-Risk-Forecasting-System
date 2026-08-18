import pandas as pd
from ml.classifier import train
def test_classifier(tmp_path):
 n=240;df=pd.DataFrame({"TransactionAmt":range(1,n+1),"TransactionDT":range(n),"amount_log1p":[2]*n,"segment_txn_24h":[1]*n,"segment_txn_1h":[1]*n,"segment_amt_24h":range(n),"missing_field_count":[0]*n,"ProductCD":["W","C"]*120,"card4":["visa","master"]*120,"card6":["credit","debit"]*120,"P_emaildomain":["gmail","yahoo"]*120,"isFraud":[1 if i%11==0 else 0 for i in range(n)]});r=train(df,str(tmp_path/"m.joblib"));assert r is not None
