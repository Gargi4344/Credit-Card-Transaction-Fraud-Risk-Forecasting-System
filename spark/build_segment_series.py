from pathlib import Path
from pyspark.sql import SparkSession, functions as F
FEATURES="data/processed/fraud_features"
OUT="data/processed/segment_daily"
def main():
 s=SparkSession.builder.appName("SegmentFraudSeries").config("spark.sql.shuffle.partitions","64").getOrCreate()
 df=s.read.parquet(FEATURES);segments=[c for c in ["ProductCD","card4"] if c in df.columns]
 if not segments: raise ValueError("No segment columns available")
 out=(df.groupBy(*(segments+["transaction_day"]))
  .agg(F.count("*").alias("transaction_count"),F.sum("isFraud").alias("fraud_count"),F.avg("TransactionAmt").alias("avg_amount"))
  .withColumn("fraud_rate",F.col("fraud_count")/F.col("transaction_count")))
 Path(OUT).parent.mkdir(parents=True,exist_ok=True);out.write.mode("overwrite").parquet(OUT);s.stop()
if __name__=="__main__": main()
