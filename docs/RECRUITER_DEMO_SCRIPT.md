# Recruiter Demo Script

## Goal
Show the project in 60 to 90 seconds with a strong engineering story.

## Suggested flow

### Shot 1: GitHub repo home
Say:
"This project is a geotechnical edge-AI prototype for underground mining risk ranking in Laverton, WA. It combines WA public geology, structures, gravity, drillholes, and regional seismicity with synthetic operational proxies where mine data is not publicly available."

### Shot 2: Repo structure
Show:
- `scripts/`
- `src/app/`
- `docs/`
- `outputs/plots/`

Say:
"The pipeline covers ingestion, feature engineering, synthetic data completion, model training, validation, and API inference."

### Shot 3: Local dashboard
Open:
- `/dashboard`

Say:
"The dashboard gives a quick engineering view of model quality and the current visual diagnostics before and after training."

### Shot 4: Visuals
Point to:
- label distribution
- correlation heatmap
- confusion matrix
- prediction confidence

Say:
"These figures help check whether the training data and model behavior are stable enough for prototype decision support."

### Shot 5: Validation
Mention:
- selected model
- macro F1
- White Reality Check

Say:
"I also added a White Reality Check to reduce the risk of overfitting through model selection. The current result suggests the selected boosting model is outperforming the baseline beyond simple data snooping."

### Shot 6: API
Show:
- `/docs`
- sample `/score` payload

Say:
"The API is designed for edge deployment. It can score incoming feature packets and return hazard type, confidence, and alert level."

## Final close
Say:
"This is still a prototype, not mine-grade validation. The next step is replacing synthetic operational proxies with site PPV, blast, and microseismic data."
