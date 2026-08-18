from pathlib import Path
from pyspark.sql import SparkSession, Window, functions as F

RAW_TX="data/raw/train_transaction.csv"
RAW_ID="data/raw/train_identity.csv"
OUT="data/processed/fraud_features"

def spark():
    return (SparkSession.builder.appName("FraudFeatureEngineering")
            .config("spark.sql.shuffle.partitions","96").getOrCreate())

def read_inputs(s):
    tx=s.read.option("header",True).option("inferSchema",True).csv(RAW_TX)
    identity=None
    if Path(RAW_ID).exists():
        identity=s.read.option("header",True).option("inferSchema",True).csv(RAW_ID)
    return tx,identity

def build_features(tx):
    required=["TransactionDT","TransactionAmt","ProductCD","isFraud"]
    missing=[c for c in required if c not in tx.columns]
    if missing: raise ValueError(f"Missing required columns: {missing}")
    df=tx.withColumn("transaction_day",F.floor(F.col("TransactionDT")/86400).cast("int"))
    df=df.withColumn("amount_log1p",F.log1p(F.col("TransactionAmt")))
    for c in ["card1","card4","card6","P_emaildomain"]:
        if c in df.columns: df=df.withColumn(c,F.coalesce(F.col(c).cast("string"),F.lit("UNKNOWN")))
    key="card1" if "card1" in df.columns else "ProductCD"
    w24=Window.partitionBy(key).orderBy(F.col("TransactionDT")).rangeBetween(-86400,0)
    w1=Window.partitionBy(key).orderBy(F.col("TransactionDT")).rangeBetween(-3600,0)
    df=(df.withColumn("segment_txn_24h",F.count("*").over(w24))
          .withColumn("segment_txn_1h",F.count("*").over(w1))
          .withColumn("segment_amt_24h",F.sum("TransactionAmt").over(w24)))
    feature_cols=[c for c in df.columns if c not in {"isFraud","TransactionID"}]
    missing_expr=sum(F.when(F.col(c).isNull(),1).otherwise(0) for c in feature_cols)
    return df.withColumn("missing_field_count",missing_expr)

def main():
    s=spark();tx,identity=read_inputs(s)
    if identity is not None and "TransactionID" in tx.columns and "TransactionID" in identity.columns:
        tx=tx.join(identity,on="TransactionID",how="left")
    out=build_features(tx);Path(OUT).parent.mkdir(parents=True,exist_ok=True)
    out.write.mode("overwrite").parquet(OUT);s.stop()

if __name__=="__main__": main()
