# Click Through Rate Prediction

A machine learning pipeline for predicting Click-Through Rate (CTR) for publication-campaign ad placements. This repository contains both the exploratory analysis and a production-ready implementation using an XGBoost ensemble model.

## Overview

The system predicts the expected CTR when placing an advertisement campaign on a specific newsletter publication. It uses a **2-fold XGBoost ensemble** approach that trains separate models for small and large audiences, capturing the different CTR dynamics across audience sizes.

### Key Features

- **2-Fold Ensemble Model**: Separate XGBoost models for small audiences (`opens < 1000`) and large audiences (`opens >= 1000`)
- **Rich Feature Engineering**: Historical CTR statistics, publication audience metrics, TF-IDF from publication tags, temporal features, and publication cluster embeddings
- **Feature Store**: Parquet-based feature store for efficient storage and retrieval
- **REST API**: FastAPI service for real-time CTR predictions
- **CLI Interface**: Command-line tools for training, evaluation, and prediction

## Experimentation

`notebooks/` contains the exploratory notebooks and scripts used for data analysis, feature engineering, and model experimentation:

| File | Description |
|------|-------------|
| `ctr_data_exploration.ipynb` | Initial data exploration and visualization |
| `data_exploration.ipynb` | Deep dive into data distributions |
| `ctr_exploration.py` | CTR patterns analysis |
| `xgboost_ctr_model.py` | Single XGBoost model experiments |
| `xgboost-2fold.py` | 2-fold ensemble model development |
| `ctr_ensemble_model.py` | Final ensemble model refinement |
| `embedding_ctr_analysis.py` | Sentence embedding analysis for publications |
| `publication_clustering.py` | K-means clustering on publication embeddings |
| `mae_by_opens_bins.py` | Error analysis by audience size bins |
| `variance_explainability.py` | Feature importance and variance analysis |

All Python scripts are [marimo](https://marimo.io) notebooks for interactive debugging and easy porting to production modules.

## Architecture

```
src/
├── api/                    # REST API layer
│   ├── app.py              # FastAPI application and endpoints
│   ├── config.py           # API configuration (env variables)
│   ├── dtos.py             # Request/response data transfer objects
│   └── service.py          # Prediction service orchestration
│
├── domain/                 # Domain layer
│   ├── entities.py         # Core entities (PlacementRecord, PredictionResult, etc.)
│   └── protocols.py        # Interfaces/protocols for dependency injection
│
├── features/               # Feature engineering
│   └── feature_engineering.py  # CTRFeatureEngineer class with all transformations
│
├── feature_store/          # Feature persistence
│   └── feature_store.py    # Parquet-based feature store implementation
│
├── models/                 # ML models
│   └── xgboost_ensemble.py # XGBoost 2-fold ensemble implementation
│
├── metrics/                # Evaluation metrics
│   └── metrics.py          # MAE, RMSE, and custom metrics
│
├── pipelines/              # ML pipelines
│   ├── config.py           # Pipeline configuration (YAML loader)
│   ├── training.py         # Training pipeline
│   ├── evaluation.py       # Evaluation pipeline with reporting
│   └── prediction.py       # Prediction pipeline
│
└── main.py                 # CLI entry point
```

### Feature Engineering

The `CTRFeatureEngineer` builds the following feature groups:

- **Campaign CTR Statistics**: Rolling window mean, std, count, and weighted CTR
- **Publication Audience Metrics**: Average/std opens, historical CTR, placement count
- **Campaign Targeting**: One-hot encoded gender, promoted item type, income/age ranges
- **Temporal Features**: Day of week, hour bucket (morning/midday/night), month
- **Content Features**: TF-IDF vectors from publication tags
- **Cluster Features**: One-hot encoded publication clusters from embeddings

## Installation

Requires Python 3.10+. Using [uv](https://github.com/astral-sh/uv) (recommended):

```bash
# Install dependencies
uv sync

# Or with pip
pip install -e .
```

## How to Run

### Configuration

All pipeline settings are configured in `pipeline_config.yml`:

```yaml
paths:
  data_dir: "data"
  output_dir: "artifacts"

model:
  max_depth: 6
  learning_rate: 0.1
  n_estimators: 200

training:
  test_split_days: 90
  audience_threshold: 1000
```

### Training

Train a new model from placement data:

```bash
# Using config file
python -m src.main train --config pipeline_config.yml

# With parameter overrides
python -m src.main train --config pipeline_config.yml --max-depth 8 --learning-rate 0.05

# With explicit paths (no config)
python -m src.main train --data-dir ./data --output-dir ./artifacts
```

**Required data files in `data/`:**
- `placements.parquet` - Placement records with CTR data
- `campaigns.parquet` - Campaign metadata
- `tags.parquet` - Publication tags
- `clusters.parquet` (optional) - Publication cluster assignments

### Evaluation

Evaluate model performance on test data:

```bash
python -m src.main evaluate --config pipeline_config.yml

# With custom artifacts directory
python -m src.main evaluate --artifacts-dir ./my_artifacts --feature-set test_features
```

### Prediction (CLI)

Generate a prediction for a single publication-campaign pair:

```bash
python -m src.main predict \
  --config pipeline_config.yml \
  --publication-id <PUBLICATION_UUID> \
  --campaign-id <CAMPAIGN_UUID>
```

### REST API

Start the prediction API server:

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

**Endpoints:**

- `POST /invoke` - Generate CTR predictions for publication-campaign pairs
- `GET /health` - Health check

**Example request:**

```bash
curl -X POST http://localhost:8000/invoke \
  -H "Content-Type: application/json" \
  -d '{
    "data": [
      {"publication_id": "uuid-1", "campaign_id": "uuid-2"},
      {"publication_id": "uuid-3", "campaign_id": "uuid-4"}
    ]
  }'
```

**Response:**

```json
{
  "predictions": [
    {"estimated_ctr": 0.0234, "model_id": "XGBoostEnsemble_threshold1000"},
    {"estimated_ctr": 0.0187, "model_id": "XGBoostEnsemble_threshold1000"}
  ],
  "execution_time_ms": 12.34,
  "response_id": "uuid"
}
```

## Environment Variables

For the API, configure via environment variables or `.env` file:

| Variable | Description | Default |
|----------|-------------|---------|
| `ARTIFACTS_DIR` | Path to model artifacts | `artifacts` |
| `FEATURE_SET_NAME` | Feature set for predictions | `prediction_features` |
