# Feature Engineering

Features are calculated only from information available at the prediction cutoff. The current helper supports:

- calendar features: month and day of year
- station-scoped lag values such as lag 1, 3, and 7
- station-scoped prior rolling means such as 3-day and 7-day windows

Every rolling window ends before the current row, and stations never share history. The helper preserves early-history `None` values; a later training pipeline must decide how to handle them and document that decision.

The final feature list must be derived from verified source columns. It must be saved with the model and reproduced exactly for manual and live predictions.
