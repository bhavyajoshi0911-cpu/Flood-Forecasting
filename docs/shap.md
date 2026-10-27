# SHAP

`backend/app/ml/explain.py` uses `shap.TreeExplainer` with a supplied trained tree model and verified feature matrix. It provides:

- global mean absolute SHAP importance
- individual feature contributions
- JSON persistence
- summary plot generation

The service validates feature names, matrix shape, and numeric values. SHAP failures are surfaced explicitly. It does not produce explanations without a real trained model and does not hardcode feature influence statements.
