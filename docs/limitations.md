# Limitations

The repository contains raw basin files and a trained model with test metrics,
but source provenance and license are not yet verified, the target is a
non-authoritative station-specific 95th-percentile research assumption, and the
raw `.basin` endpoint performs only row/header reading. Metrics and SHAP
summaries describe these research artifacts only; they are not validated
real-world performance or official flood-event evidence.

Live predictions are unavailable unless the configured provider supplies every
required feature with validated station, timestamp, units, and values. Manual
predictions also require the complete model feature vector. Missing features
are rejected rather than filled from held-out test data. Historical target
labels are not predictions, and the application does not fabricate prediction
history or alert records.

The application can separately retrieve the Central Water Commission Flood
Forecasting portal's public list of stations reported above warning level.
That feed does not include source timestamps, units, station names, or basin
mapping, and is not used as input to the local model. Verify station location
and observation timing with the source before interpreting an entry.

Potential limitations after data integration include station mismatch, missing observations, uncertain event labels, measurement errors, class imbalance, distribution shift, provider outages, stale readings, unsupported feature gaps, and threshold uncertainty. These must be reported with the relevant dataset, model, and API status.
