from pathlib import Path
import joblib,pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score,average_precision_score,classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.ensemble import HistGradientBoostingClassifier
NUM=["TransactionAmt","TransactionDT","amount_log1p","segment_txn_24h","segment_txn_1h","segment_amt_24h","missing_field_count"]
CAT=["ProductCD","card4","card6","P_emaildomain"]
def train(df,path="artifacts/fraud_classifier.joblib"):
 cols=[c for c in NUM+CAT if c in df.columns];X=df[cols];y=df.isFraud.astype(int)
 Xtr,Xte,ytr,yte=train_test_split(X,y,test_size=.2,stratify=y,random_state=42)
 num=[c for c in NUM if c in X];cat=[c for c in CAT if c in X]
 pre=ColumnTransformer([("num",SimpleImputer(strategy="median"),num),("cat",Pipeline([("imp",SimpleImputer(strategy="most_frequent")),("enc",OrdinalEncoder(handle_unknown="use_encoded_value",unknown_value=-1))]),cat)])
 model=HistGradientBoostingClassifier(max_iter=250,learning_rate=.08,max_leaf_nodes=31,l2_regularization=1,random_state=42)
 pipe=Pipeline([("pre",pre),("model",model)]);pipe.fit(Xtr,ytr);prob=pipe.predict_proba(Xte)[:,1]
 print("ROC-AUC",roc_auc_score(yte,prob));print("PR-AUC",average_precision_score(yte,prob));print(classification_report(yte,(prob>=.5).astype(int),digits=4))
 Path(path).parent.mkdir(parents=True,exist_ok=True);joblib.dump(pipe,path);return pipe
if __name__=="__main__": train(pd.read_parquet("data/processed/fraud_features"))
