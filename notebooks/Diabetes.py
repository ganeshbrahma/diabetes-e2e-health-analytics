# Databricks notebook source
# MAGIC %md
# MAGIC # Configure access to Blob (uses the secret key)

# COMMAND ----------

# === CONFIG ===
STORAGE_ACCOUNT = "sthealthdemogk" 
CONTAINER = "raw"

# Use the secret created earlier
spark.conf.set(f"fs.azure.account.key.{STORAGE_ACCOUNT}.blob.core.windows.net",
               dbutils.secrets.get(scope="azure", key="blob-key"))

# URIs
root_uri = f"wasbs://{CONTAINER}@{STORAGE_ACCOUNT}.blob.core.windows.net"
src_dir  = f"{root_uri}/diabetes"
bronze   = f"{root_uri}/lake/bronze/diabetes"
silver   = f"{root_uri}/lake/silver/diabetes"
gold     = f"{root_uri}/lake/gold/diabetes"

catalog = "hive_metastore"   # default metastore
db      = "health"           # database (schema) name


# COMMAND ----------

# MAGIC %md
# MAGIC # Create a database (schema) for the tables

# COMMAND ----------

spark.sql(f"CREATE DATABASE IF NOT EXISTS {catalog}.{db}")
spark.sql(f"USE {catalog}.{db}")


# COMMAND ----------

# MAGIC %md
# MAGIC # BRONZE — Read all CSVs from Blob and write Delta

# COMMAND ----------

from pyspark.sql import functions as F

# Read every daily drop copy with ADF
src_glob = f"{src_dir}/year=*/month=*/day=*/diabetes_*.csv"

bronze_df = (spark.read
             .option("header", True)
             .option("inferSchema", True)
             .csv(src_glob)
             .withColumn("ingest_ts", F.current_timestamp()))

# Write Bronze as Delta (append-safe)
(bronze_df
 .write
 .mode("append")
 .format("delta")
 .save(bronze))

# Register Delta table if not exists
spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db}.diabetes_bronze
USING DELTA
LOCATION '{bronze}'
""")

#display(spark.table(f"{db}.diabetes_bronze").limit(5))

display(spark.sql(f"select * from {db}.diabetes_bronze"))


# COMMAND ----------

# MAGIC %md
# MAGIC # SILVER — Clean types, handle zeros as missing, basic QC

# COMMAND ----------

bronze_tbl = spark.table(f"{db}.diabetes_bronze")

numeric_cols = ["pregnancies","glucose","bloodpressure","skinthickness",
                "insulin","bmi","diabetespedigreefunction","age","outcome"]

# Cast columns explicitly (schema can vary per CSV)
silver_df = (bronze_tbl
    .select(
        F.col("pregnancies").cast("int"),
        F.col("glucose").cast("int"),
        F.col("bloodpressure").cast("int"),
        F.col("skinthickness").cast("int"),
        F.col("insulin").cast("int"),
        F.col("bmi").cast("double"),
        F.col("diabetespedigreefunction").cast("double"),
        F.col("age").cast("int"),
        F.col("outcome").cast("int"),
        F.col("ingested_at").cast("timestamp").alias("ingested_at") if "ingested_at" in bronze_tbl.columns else F.current_timestamp().alias("ingested_at"),
        F.col("ingest_ts"))
    )

# In this dataset, zeros can indicate missing for some fields; convert to null
zero_to_null = {
    "glucose", "bloodpressure", "skinthickness", "insulin", "bmi"
}
for c in zero_to_null:
    if c in silver_df.columns:
        silver_df = silver_df.withColumn(c, F.when(F.col(c)==0, None).otherwise(F.col(c)))

# Basic QC: drop rows missing target or key predictors
silver_df = silver_df.dropna(subset=["glucose","bmi","bloodpressure","outcome"])

# Write SILVER as Delta (overwrite partition for demo)
(silver_df.write
    .mode("overwrite")
    .format("delta")
    .save(silver))

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db}.diabetes_silver
USING DELTA
LOCATION '{silver}'
""")

#display(spark.table(f"{db}.diabetes_silver").summary())

display(spark.sql(f"select count(*) from {db}.diabetes_silver"))


# COMMAND ----------

# MAGIC %md
# MAGIC # GOLD — Feature engineering view/table (ready for BI/ML)

# COMMAND ----------

sdf = spark.table(f"{db}.diabetes_silver")

gold_df = (sdf
    .withColumn("bmi_band",
        F.when(F.col("bmi") < 18.5, "under")
         .when((F.col("bmi") >= 18.5) & (F.col("bmi") < 25), "normal")
         .when((F.col("bmi") >= 25) & (F.col("bmi") < 30), "over")
         .otherwise("obese"))
    .withColumn("age_band",
        F.when(F.col("age") < 30, "<30")
         .when((F.col("age") >= 30) & (F.col("age") < 45), "30-44")
         .when((F.col("age") >= 45) & (F.col("age") < 60), "45-59")
         .otherwise("60+"))
)

(gold_df.write
   .mode("overwrite")
   .format("delta")
   .save(gold))

spark.sql(f"""
CREATE TABLE IF NOT EXISTS {db}.diabetes_gold
USING DELTA
LOCATION '{gold}'
""")

display(spark.table(f"{db}.diabetes_gold"))


# COMMAND ----------

from pyspark.ml.feature import VectorAssembler, StringIndexer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml import Pipeline
from pyspark.ml.evaluation import BinaryClassificationEvaluator

data = spark.table(f"{db}.diabetes_silver").na.drop()

features = ["pregnancies","glucose","bloodpressure","skinthickness","insulin","bmi","diabetespedigreefunction","age"]
assembler = VectorAssembler(inputCols=features, outputCol="features")
lr = LogisticRegression(featuresCol="features", labelCol="outcome", maxIter=50)

pipe = Pipeline(stages=[assembler, lr])
train, test = data.randomSplit([0.8, 0.2], seed=42)
model = pipe.fit(train)
pred  = model.transform(test)

auc = BinaryClassificationEvaluator(labelCol="outcome").evaluate(pred)
print(f"AUC: {auc:.3f}")
