# Databricks notebook source
# Databricks: install into the cluster's Python env
%pip install shap
#%restart_python

# COMMAND ----------

catalog = "hive_metastore"
db      = "health"
raw     = "wasbs://raw@sthealthdemogk.blob.core.windows.net"  # your storage
bronze  = f"{catalog}.{db}.diabetes_ds_bronze"
silver  = f"{catalog}.{db}.diabetes_ds_silver"
gold    = f"{catalog}.{db}.diabetes_ds_gold"
gold_scored = f"{catalog}.{db}.diabetes_scored"
monitor_tbl = f"{catalog}.{db}.model_monitoring"

source_path = f"{raw}/lake/gold/diabetes"  # where ADF writes CSV/Delta
seed = 42


# COMMAND ----------

# MAGIC %md
# MAGIC # Land raw data into Bronze Delta (schema-on-read)

# COMMAND ----------

from pyspark.sql.functions import current_timestamp, input_file_name

spark.sql(f"CREATE DATABASE IF NOT EXISTS {db}")

df = (spark.read
      .format("delta")        # if your ADF already writes CSV, use .format("csv").option("header", True)
      .load(source_path)
      .withColumn("ingested_at", current_timestamp())
      .withColumn("source_file", input_file_name()))

(df.write
   .mode("overwrite")
   .format("delta")
   .saveAsTable(bronze))


# COMMAND ----------

# MAGIC %md
# MAGIC # Clean, impute & standardize to Silver

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.table(bronze)

# Basic QC
cols = ["pregnancies","glucose","bloodpressure","skinthickness",
        "insulin","bmi","diabetespedigreefunction","age","outcome"]
df = df.select(*cols, "ingested_at")

# Treat zero as missing for medical measures commonly zero filled
for c in ["glucose","bloodpressure","skinthickness","insulin","bmi"]:
    df = df.withColumn(c, F.when(F.col(c)==0, None).otherwise(F.col(c)))

# Simple imputations (median)
impute_expr = {c: "median" for c in ["glucose","bloodpressure","skinthickness","insulin","bmi"]}
# quick Spark approx: fill with percentiles
stats = df.agg(*[F.expr(f"percentile_approx({c}, 0.5)") .alias(c) for c in impute_expr]).collect()[0].asDict()
df = df.fillna(stats)

# Banding
df = (df
      .withColumn("bmi_band", F.when(F.col("bmi")<25,"normal")
                                .when(F.col("bmi")<30,"over")
                                .otherwise("obese"))
      .withColumn("age_band", F.when(F.col("age")<30,"lt30")
                               .when(F.col("age")<45,"30-44")
                               .when(F.col("age")<60,"45-59")
                               .otherwise("60+")))

df.write.mode("overwrite").format("delta").saveAsTable(silver)


# COMMAND ----------

# MAGIC %md
# MAGIC # Assemble model features (Gold)

# COMMAND ----------

from pyspark.sql import functions as F

df = spark.table(silver)

feature_cols = ["pregnancies","glucose","bloodpressure","skinthickness",
                "insulin","bmi","diabetespedigreefunction","age"]
label_col = "outcome"

# Add a synthetic id to support monitoring/joins
df = df.withColumn("patient_id", F.monotonically_increasing_id())

df.select(["patient_id"] + feature_cols + [label_col, "bmi_band","age_band","ingested_at"]) \
  .write.mode("overwrite").format("delta").saveAsTable(gold)


# COMMAND ----------

# MAGIC %md
# MAGIC # AutoML-style tuning & tracking; register best model
# MAGIC ### Train/tune multiple algorithms with MLflow, pick best by ROC-AUC; push to Model Registry.

# COMMAND ----------

import mlflow
from mlflow.tracking import MlflowClient

# Ensure a stable seed (in case it's not defined)
seed = 42

# Use Databricks tracking
mlflow.set_tracking_uri("databricks")

# Build an experiment path under user folder (which exists)
user = spark.sql("SELECT current_user()").first()[0]        
EXP_PATH = f"/Users/{user}/health_diabetes_experiment"

# Create the experiment if missing, then get its id
client = MlflowClient()
exp = mlflow.get_experiment_by_name(EXP_PATH)
if exp is None:
    exp_id = client.create_experiment(EXP_PATH)
else:
    exp_id = exp.experiment_id

print("Using MLflow experiment:", EXP_PATH, " (id:", exp_id, ")")


# COMMAND ----------

import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
import pandas as pd

tbl = spark.table(gold).toPandas()
X = tbl[["pregnancies","glucose","bloodpressure","skinthickness","insulin","bmi","diabetespedigreefunction","age"]]
y = tbl["outcome"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)

mlflow.set_experiment("/Shared/health_diabetes")  # create once

candidates = [
  ("logreg", Pipeline([("scale", StandardScaler()), 
                       ("clf", LogisticRegression(max_iter=200, class_weight="balanced"))]),
            {}),

  ("xgb", XGBClassifier(
         n_estimators=300, max_depth=4, learning_rate=0.08,
         subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
         eval_metric="logloss", random_state=seed, n_jobs=-1), {})
]

best = None
best_auc = -1

for name, model, params in candidates:
    with mlflow.start_run(run_name=name, experiment_id=exp_id):
        model.fit(X_train, y_train)
        proba = model.predict_proba(X_test)[:,1]
        auc = roc_auc_score(y_test, proba)
        ap  = average_precision_score(y_test, proba)

        mlflow.log_metric("roc_auc", auc)
        mlflow.log_metric("pr_auc", ap)
        mlflow.log_params({"algo": name, **params})
        mlflow.sklearn.log_model(model, artifact_path="model")

        if auc > best_auc:
            best_auc, best = auc, (name, model)

# Register best
name, model = best
with mlflow.start_run(run_name=f"register_{name}", experiment_id=exp_id):
    mlflow.sklearn.log_model(
        model, "model",
        registered_model_name="diabetes_classification"
    )


# COMMAND ----------

# MAGIC %md
# MAGIC # Holdout evaluation, calibration & SHAP
# MAGIC ### Produce business ready artifacts: ROC/PR curves, confusion matrix at chosen threshold, calibration, SHAP global + local explanations; log to MLflow.

# COMMAND ----------

import shap, numpy as np, matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import RocCurveDisplay, PrecisionRecallDisplay, confusion_matrix

# Load best from registry (Staging)
from mlflow.tracking import MlflowClient
client = MlflowClient()
version = client.get_latest_versions("diabetes_classification", stages=["None","Staging","Production"])[0].version
model = mlflow.sklearn.load_model(model_uri=f"models:/diabetes_classification/{version}")

# Reuse train/test from previous notebook or reload/reshuffle deterministically
# ... X_test, y_test as before ...

proba = model.predict_proba(X_test)[:,1]
thresh = 0.5  # adjust based on PR curve / business cost
pred  = (proba >= thresh).astype(int)

with mlflow.start_run(run_name="evaluation"):
    # ROC + PR
    fig, ax = plt.subplots(); RocCurveDisplay.from_predictions(y_test, proba, ax=ax); mlflow.log_figure(fig,"roc.png"); plt.close(fig)
    fig, ax = plt.subplots(); PrecisionRecallDisplay.from_predictions(y_test, proba, ax=ax); mlflow.log_figure(fig,"pr.png"); plt.close(fig)

    # Confusion matrix
    cm = confusion_matrix(y_test, pred)
    np.savetxt("/dbfs/tmp/cm.csv", cm, delimiter=",", fmt="%d"); mlflow.log_artifact("/dbfs/tmp/cm.csv", "eval")

    # Calibration
    prob_true, prob_pred = calibration_curve(y_test, proba, n_bins=10)
    fig, ax = plt.subplots(); ax.plot(prob_pred, prob_true, marker="o"); ax.plot([0,1],[0,1],"--"); ax.set_title("Calibration"); mlflow.log_figure(fig,"calibration.png"); plt.close(fig)

    # SHAP (tree models: TreeExplainer; for LR: KernelExplainer)
    explainer = shap.Explainer(model)
    shap_values = explainer(X_test)
    shap.plots.beeswarm(shap_values, show=False); plt.tight_layout(); mlflow.log_figure(plt.gcf(),"shap_beeswarm.png"); plt.close()
    shap.plots.bar(shap_values, show=False); mlflow.log_figure(plt.gcf(),"shap_importance.png"); plt.close()


# COMMAND ----------

# MAGIC %md
# MAGIC # Score latest Gold, write scored Delta for BI & ops
# MAGIC ### Load Production model → score all patients → write health.diabetes_scored with probabilities & bands.

# COMMAND ----------

import mlflow, pyspark.sql.functions as F

# Load Production model (move your best to Production in Model Registry UI or via API)
model = mlflow.sklearn.load_model("models:/diabetes_classification/Production")

sdf = spark.table(gold)
features = ["pregnancies","glucose","bloodpressure","skinthickness",
            "insulin","bmi","diabetespedigreefunction","age"]

pdf = sdf.select(["patient_id"] + features + ["bmi_band","age_band"]).toPandas()
pdf["score"] = model.predict_proba(pdf[features])[:,1]

scored = (spark.createDataFrame(pdf)
          .withColumn("score_ts", F.current_timestamp())
          .withColumn("risk_band", F.when(F.col("score")>=0.7,"High")
                                    .when(F.col("score")>=0.4,"Medium")
                                    .otherwise("Low")))

(scored.write.mode("overwrite").format("delta").saveAsTable(gold_scored))


# COMMAND ----------

# MAGIC %md
# MAGIC # Data & model drift + production KPIs
# MAGIC ### Compute PSI (Population Stability Index) for key features, monitor score drift & threshold performance; store in Delta for dashboards/alerts.

# COMMAND ----------

from pyspark.sql import functions as F
import numpy as np

scored = spark.table(gold_scored)

# Example: weekly aggregates & drift vs baseline
baseline = scored.where("date(score_ts) = (select min(date(score_ts)) from {gold_scored})")
latest   = scored.where("date(score_ts) = (select max(date(score_ts)) from {gold_scored})")

def psi_bins(col):
    # simple PSI by equal bins
    return f"""
    WITH b AS (
      SELECT ntile(10) OVER (ORDER BY {col}) bin, count(*) as n FROM {gold_scored} WHERE date(score_ts)=(SELECT min(date(score_ts)) FROM {gold_scored}) GROUP BY 1
    ),
    l AS (
      SELECT ntile(10) OVER (ORDER BY {col}) bin, count(*) as n FROM {gold_scored} WHERE date(score_ts)=(SELECT max(date(score_ts)) FROM {gold_scored}) GROUP BY 1
    )
    SELECT COALESCE(SUM( (l.n/t1.tot) - (b.n/t0.tot) ) * ln( (l.n/t1.tot)/NULLIF((b.n/t0.tot),0) ),0) as psi
    FROM b JOIN l USING(bin),
         (SELECT COUNT(*) tot FROM {gold_scored} WHERE date(score_ts)=(SELECT min(date(score_ts)) FROM {gold_scored})) t0,
         (SELECT COUNT(*) tot FROM {gold_scored} WHERE date(score_ts)=(SELECT max(date(score_ts)) FROM {gold_scored})) t1
    """
# You can compute for score, glucose, bmi, age, etc., and insert into monitor table.

spark.sql(f"CREATE TABLE IF NOT EXISTS {monitor_tbl} (metric string, value double, ts timestamp)")
psi_score = spark.sql(psi_bins("score")).first()["psi"]
spark.sql(f"INSERT INTO {monitor_tbl} VALUES ('psi_score', {psi_score}, current_timestamp())")
