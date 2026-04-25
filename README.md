# mining-risk-intelligence

Geotechnical and edge-AI prototype for mining ground-risk ranking in Western Australia. The project builds a public regional context layer from WA government and Geoscience Australia datasets, adds synthetic operational proxies where mine data is unavailable, trains a tabular ML model, and serves predictions through a FastAPI endpoint.

## What it does
- ingests WA public geology, structure, gravity, drillhole, mineralisation, and seismic context
- engineers spatial features such as structure proximity, drillhole density, mineral proximity, and gravity values
- generates synthetic PPV, stress, blast, and rock-mass proxy fields to complete the prototype pipeline
- trains and compares lightweight classification models
- visualises the data before and after training
- serves hazard predictions through an API

## Repo structure
```text
mining-risk-intelligence/
  configs/
    request.json
  data/
    raw/
      public/
      seismic/
      examples/
  docs/
    RESEARCH_BACKED_ML.md
    SOURCES_AND_CITATIONS.md
  outputs/
  scripts/
    feature_engineering.py
    ga_xml_to_csv.py
    gravity_sampling.py
    merge_training_data.py
    sarig_to_csv.py
    synthetic_operational_data.py
    train.py
    visualize_pipeline.py
    wa_api_ingest.py
  src/
    app/
      main.py
      inference.py
      training.py
      schemas.py
      ...
  README.md
  requirements.txt
```

## Data used
### Public datasets
- `DMIRS-015` State Linear Structures
- `DMIRS-038` Interpreted Bedrock Geology 1:100k
- `DMIRS-070` Gravity 400 m of WA
- `DMIRS-046` Mineral Exploration Drillholes
- `MINEDEX`
- `WAMEX`
- `Geoscience Australia` earthquake catalogue / XML event records

### Public services and APIs
- DMIRS / SLIP ArcGIS REST services for structures, geology, and gravity
- GeoVIEW WA and WA Data Catalogue
- Geoscience Australia earthquake records

### Prototype-only synthetic fields
- `synthetic_ppv_mm_s`
- `synthetic_charge_per_delay_kg`
- `synthetic_dominant_frequency_hz`
- `synthetic_inferred_rqd`
- `synthetic_inferred_gsi`
- `synthetic_sigma_v_mpa`
- `synthetic_sigma_h_mpa`
- `synthetic_seismic_magnitude`
- `synthetic_seismic_depth_m`
- `target_label`

## Install
```bash
pip install -r requirements.txt
```

## Pipeline
### 1. Build public or engineered feature tables
```bash
python scripts/feature_engineering.py --geology "data/raw/public/1_100_000_Interpreted_Bedrock_Geology.shp" --minerals "data/raw/public/Mines_and_Mineral_Deposits__MINEDEX_.shp" --drillholes "data/raw/public/Mineral_Exploration_Drillholes__open_file_.shp" --output data/geology_features.csv
python scripts/gravity_sampling.py --input data/geology_features.csv --output data/geology_features_with_gravity.csv --lat-col centroid_lat --lon-col centroid_lon
```

### 2. Convert GA seismic XML
```bash
python scripts/ga_xml_to_csv.py --input "C:\Users\pored\Downloads\ga2026hzufif.xml" --output data/regional_seismic_events.csv
```

### 3. Generate synthetic operational data
```bash
python scripts/synthetic_operational_data.py --output data/synthetic_operational_data.csv --rows 5000
```

### 4. Build the final training table
```bash
python scripts/merge_training_data.py --public-features data/geology_features_with_gravity.csv --synthetic-operational data/synthetic_operational_data.csv --seismic data/regional_seismic_events.csv --output data/final_training_table.csv
```

### 5. Visualise before training
```bash
python scripts/visualize_pipeline.py --training-table data/final_training_table.csv --output-dir outputs/plots
```

### 6. Train models
```bash
python scripts/train.py
```

### 7. Visualise test performance
```bash
python scripts/visualize_pipeline.py --training-table data/final_training_table.csv --predictions data/test_predictions.csv --feature-importance data/feature_importance.csv --output-dir outputs/plots
```

### 8. Run the API
```bash
uvicorn src.app.main:app --reload
```

### 9. Score a sample event
```bash
curl -X POST "http://127.0.0.1:8000/score" -H "Content-Type: application/json" --data @configs/request.json
```

## Model stack
- Logistic Regression baseline
- Random Forest
- HistGradientBoostingClassifier

Model selection uses `macro_f1` to avoid over-favoring the majority label.

## API
### `GET /health`
Returns service health.

### `POST /score`
Consumes merged-table style features and returns:
- `Hazard Type`
- `Risk Score`
- `Alert Level`
- `Failure Mechanism`
- `Confidence Level`
- `Class Probabilities`
- `Top Drivers`

## Notes
- Public datasets provide the regional context layer only.
- Synthetic fields are placeholders for missing mine-operational data.
- Replace synthetic PPV, blast, stress, and rock-mass fields with site data before treating the output as operationally valid.
- Dataset and research references are documented in `docs/SOURCES_AND_CITATIONS.md` and `docs/RESEARCH_BACKED_ML.md`.

## GitHub publish
I prepared this folder as a GitHub-ready repository named `mining-risk-intelligence`.
If the remote repository does not exist yet, create an empty repo at:
- `https://github.com/jeevanreddy23/mining-risk-intelligence`

Then publish from this folder:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/jeevanreddy23/mining-risk-intelligence.git
git push -u origin main
```
