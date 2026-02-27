# ML Pipeline for Decision Support & Analytics
I designed and implemented an end-to-end machine learning pipeline to support data-driven decision-making using real-world historical datasets. The system ingests raw data, performs cleaning and preprocessing, applies feature engineering, trains and evaluates machine learning models, and exposes predictions through a production-style REST API.

Recommended runtime: Python `3.11` or `3.12` (Windows + Python `3.14` may appear to freeze while compiling scientific dependencies from source).

Project location: `ml-pipeline/`

Quick run:
- Activate env: `& ".\ml-pipeline\.venv\Scripts\Activate.ps1"`
- Start API/UI: `uvicorn app.main:app --reload --port 8000` (from `ml-pipeline`)
- Open UI: `http://127.0.0.1:8000/`

The detailed docs (API, demo datasets, explainability, and run comparison workflow) are in `ml-pipeline/README.md`.
