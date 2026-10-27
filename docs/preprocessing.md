# Preprocessing

The preprocessing layer currently provides explicit, source-agnostic operations:

- require timezone-aware ISO-8601 timestamps and normalize them to UTC
- coerce documented numeric measurements to floating point
- reject non-numeric values and missing station identifiers
- remove repeated station-time rows deterministically and report the count
- report missing calendar dates without filling them
- require units in dataset metadata before scientific processing

Raw files remain unchanged in `data/raw/`. Cleaning and any unit conversion must be recorded in metadata before implementation against a real source. Missing values are not silently imputed in the current phase.

Run the current quality report from the project root:

```powershell
python scripts\validate_dataset.py
```

With no configured CSV, the expected status is `not_configured`.
