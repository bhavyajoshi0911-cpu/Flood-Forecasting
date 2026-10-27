# XGBoost

`backend/app/ml/train.py` contains the configurable binary XGBoost training service. It accepts a previously constructed chronological split and validates:

- non-empty train, validation, and test partitions
- complete numeric features
- binary next-day targets
- both target classes in training data
- model, dataset, and feature version identifiers

A successful real-data training run writes a model JSON file, feature configuration JSON, and metadata JSON under `models/`. Metadata records parameters and training, validation, and test periods.

No model is currently available because no real labelled dataset has been configured. Metrics must be calculated only from the held-out chronological test partition.
