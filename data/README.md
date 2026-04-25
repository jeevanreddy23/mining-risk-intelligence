# Data Inventory

## Raw public data
Store downloaded WA datasets in `data/raw/public/`.

Current included examples:
- interpreted geology
- MINEDEX
- mineralisation sites
- exploration drillholes
- magnetic survey coverage
- WAMEX report index

## Raw seismic data
Store GA XML files and other regional seismic inputs in `data/raw/seismic/`.

## Derived data
Generated CSV files such as:
- `data/geology_features.csv`
- `data/geology_features_with_gravity.csv`
- `data/regional_seismic_events.csv`
- `data/synthetic_operational_data.csv`
- `data/final_training_table.csv`

These are intentionally ignored by git unless you choose to commit sample outputs.
