# Diabetic Patient Insights – End-to-End Healthcare Analytics

**Stack:** Azure Databricks (Delta), MLflow, scikit-learn/XGBoost, Power BI (Service), Azure Data Factory (optional)

This project takes raw patient vitals → curated Delta tables → interactive Power BI dashboard → ML model to predict diabetes risk.

## Highlights
- 🧱 Bronze → Silver → Gold Lakehouse modeling on Databricks
- 📊 Power BI dashboard (age, BMI, glucose, prevalence) with slicers and metrics
- 🤖 ML training with MLflow (LogReg vs XGBoost), best model: XGBoost (highest ROC-AUC)
- 🔁 Reproducible notebooks + requirements


## Results
- **724 patients**, **34.39% positive** prevalence
- **Avg BMI = 32.47**, **Avg Glucose = 121.88**
- Obese group shows the **highest diabetes prevalence (~45%)**
- Best model: **XGBoost** with strong ROC-AUC/PR-AUC; registered via MLflow

## Run locally (ML only)
```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
# use notebooks/04_train_diabetes_classifier.py logic as a script or in a notebook


