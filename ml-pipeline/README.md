# ML Pipeline for Decision Support & Analytics

Production-oriented, modular ML pipeline with FastAPI endpoints for ingestion, training, evaluation, inference, and experiment tracking.

## Features
- CSV ingestion with schema/type validation
- Preprocessing: missing value handling, outlier clipping, scaling, categorical encoding
- Feature engineering (`numeric_sum`, `numeric_mean`, `missing_count`)
- Multi-model training and comparison (Linear/Logistic, Random Forest, Gradient Boosting, optional XGBoost)
- Cross-validation + validation/test metrics
- Built-in frontend (`/`) for upload, train, evaluate, predict, and experiment history
- Run comparison UI (select Run A / Run B and compare metrics visually)
- Demo dataset selector + one-click latest prediction payload loader
- Local model registry and experiment metadata snapshots
- Token-based API authentication
- Prediction logging + lightweight drift checks
- Explainability summaries (global feature importance + local per-record contribution hints)
- Dockerized deployment

## Project Structure
```text
ml-pipeline/
+-- app/
�   +-- api/
�   +-- core/
�   +-- models/
�   +-- pipelines/
�   +-- utils/
+-- data/
+-- experiments/
+-- notebooks/
+-- tests/
+-- Dockerfile
+-- requirements.txt
+-- README.md
```

## Setup
Use Python `3.11` or `3.12` for this project. On Windows, Python `3.14` can trigger source builds for `pandas`/`scikit-learn`, causing very high CPU usage and apparent freezing during `pip install`.

```bash
cd ml-pipeline
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Set environment variables (optional defaults shown):
- `API_TOKEN=dev-token`
- `APP_ENV=dev`
- `LOG_LEVEL=INFO`
- `RANDOM_SEED=42`

Run API:
```bash
uvicorn app.main:app --reload --port 8000
```

Open frontend UI:
- `http://127.0.0.1:8000/`
- API docs remain at `http://127.0.0.1:8000/docs`

## Frontend Quick Start
From the UI (`/`):
- Select a demo dataset from **Demo Dataset** (or upload your own CSV)
- Set `target_column` and click **Run Pipeline**
- Use **Evaluate Latest** to refresh metrics + explainability panels
- Use **Refresh Runs** and **Compare Selected** to compare two runs visually
- Use **Load Latest Payload** to auto-fill the predict JSON box

## Authentication
Every protected endpoint requires:
- Header: `Authorization: Bearer <API_TOKEN>`

## API Endpoints
- `POST /data/upload` - Upload CSV and inspect schema
- `POST /pipeline/run` - Run end-to-end pipeline
- `GET /pipeline/status` - Pipeline state/status
- `POST /model/train` - Trigger model training
- `GET /model/evaluate` - Retrieve latest model metrics
- `POST /model/predict` - Predict from JSON records
- `GET /experiments` - List tracked experiments
- `GET /experiments/{run_id}` - Get one run's full metadata (metrics + explainability)
- `GET /predictions/latest` - Retrieve latest prediction payload for quick re-use

### Explainability in Responses
- Training/evaluation responses now include `explainability.global_feature_importance`
- Prediction responses include `explainability.local` top feature contributions per record
- Values are SHAP-style approximations derived from model coefficients/importances

## Demo Data Included
Located in `data/raw/`:
- `sample_classification.csv`
- `demo_classification.csv`
- `demo_regression.csv`
- `demo_finance_classification.csv`
- `demo_predict_records.json`
- `demo_finance_predict_records.json`

Use target column:
- Classification demos: `target`
- Regression demo: `risk_score`

## Example Flow
1. Upload data:
```bash
curl -X POST "http://localhost:8000/data/upload" \
  -H "Authorization: Bearer dev-token" \
  -F "file=@data/raw/sample_classification.csv" \
  -F "target_column=target"
```

2. Run pipeline:
```bash
curl -X POST "http://localhost:8000/pipeline/run" \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{
    "target_column": "target",
    "task_type": "classification",
    "models": ["logistic_regression", "random_forest"],
    "test_size": 0.2,
    "val_size": 0.1
  }'
```

3. Predict:
```bash
curl -X POST "http://localhost:8000/model/predict" \
  -H "Authorization: Bearer dev-token" \
  -H "Content-Type: application/json" \
  -d '{"records": [{"feature1": 5.1, "feature2": 3.5, "feature3": 1.4, "feature4": 0.2}]}'
```

4. Get one run details:
```bash
curl -X GET "http://localhost:8000/experiments/<run_id>" \
  -H "Authorization: Bearer dev-token"
```

5. Load latest prediction payload:
```bash
curl -X GET "http://localhost:8000/predictions/latest" \
  -H "Authorization: Bearer dev-token"
```

## Testing
```bash
pytest -q
```

## Reproducibility
Each run writes to `experiments/<run_id>/`:
- `model.joblib`
- `metrics.json`
- `config.json`
- `schema.json`
- `metadata.json`

`experiments/model_store/latest_model.joblib` and `latest_metadata.json` point to the latest approved model.

## Docker
```bash
docker build -t ml-pipeline .
docker run -p 8000:8000 -e API_TOKEN=dev-token ml-pipeline
```
