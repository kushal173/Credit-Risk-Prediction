# CreditLens — Credit Risk Prediction

CreditLens is a credit-risk prediction application with a Streamlit user interface and a FastAPI REST API. It uses a previously trained Home Credit default-risk model and the saved preprocessing artifacts stored in `credit1_risk_model.pkl`. The API loads the saved artifact; it does not retrain the model during prediction.

## Features

- Streamlit interface for submitting applicant information
- FastAPI endpoint for prediction
- Input validation and preprocessing using saved artifacts
- JSON prediction responses with probability, classification, risk tier, and recommendation
- Interactive API documentation through Swagger UI
- Docker Compose configuration for running the API and UI as separate services

## Project structure

| File | Purpose |
|---|---|
| `app.py` | Streamlit interface; sends applicant data to the API over HTTP |
| `api.py` | FastAPI routes, request validation, preprocessing, and predictions |
| `credit1_risk_model.pkl` | Saved model, fitted preprocessing objects, feature schema, and threshold |
| `Credit_Risk_Assessment_F.ipynb` | Notebook for model exploration, training, evaluation, and artifact creation |
| `requirements.txt` | Python dependencies |
| `Dockerfile` | Container image definition for the application |
| `docker-compose.yml` | Configuration for the API and Streamlit services |

## Requirements

- Python 3.12 for local development
- Docker Desktop with Docker Compose for containerized use
- The `credit1_risk_model.pkl` artifact in the project directory

Use trusted model artifacts only. Pickle/joblib files can execute code when loaded.

## Run locally on Windows

Open Command Prompt in the project directory.

### 1. Create a virtual environment and install dependencies

```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

### 2. Start the FastAPI service

In Terminal 1, with the virtual environment activated:

```bat
python -m uvicorn api:app --reload --host 127.0.0.1 --port 8000
```

Useful API URLs:

- Health check: http://127.0.0.1:8000/health
- Interactive API documentation: http://127.0.0.1:8000/docs
- OpenAPI schema: http://127.0.0.1:8000/openapi.json

Keep this terminal running.

### 3. Start the Streamlit interface

Open Terminal 2 in the same project directory and activate the same virtual environment:

```bat
.venv\Scripts\activate
python -m streamlit run app.py
```

Open http://localhost:8501 in your browser.

By default, the Streamlit application sends requests to `http://127.0.0.1:8000`. To configure a different API address, set the `CREDIT_RISK_API_URL` environment variable before starting Streamlit.

## Example API request

With the API running, send a JSON request to `POST /predict`. This example uses `curl` syntax supported by modern Windows installations:

```bat
curl.exe -X POST "http://127.0.0.1:8000/predict" ^
  -H "Content-Type: application/json" ^
  -d "{\"amt_income_total\":180000,\"amt_credit\":500000,\"amt_annuity\":25000,\"amt_goods_price\":450000,\"age_years\":32,\"employment_years\":5,\"cnt_children\":1,\"cnt_fam_members\":3,\"code_gender\":\"F\",\"name_contract_type\":\"Cash loans\",\"flag_own_car\":\"Y\",\"flag_own_realty\":\"Y\",\"name_income_type\":\"Working\",\"name_education_type\":\"Higher education\",\"name_family_status\":\"Married\",\"name_housing_type\":\"House / apartment\",\"ext_source_1\":0.6,\"ext_source_2\":0.5,\"ext_source_3\":0.7}"
```

The request fields and validation rules are defined by `api.py`. Fields may be omitted or set to `null` where the API permits it; saved preprocessing objects handle supported missing values. When supplied, annual income must be greater than zero, and external-source scores must be between 0 and 1.

## API endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/` | Service information |
| `GET` | `/health` | Health and model-loaded status |
| `GET` | `/metadata` | Model threshold and expected feature count |
| `POST` | `/predict` | Validate applicant data and return a prediction |

Use `/docs` to inspect the request schema and try the endpoints interactively.

The prediction response includes a default probability, threshold-based prediction, risk tier, recommendation, supplied-field count, and processed feature count.

## Run with Docker Compose

Ensure Docker Desktop is running. From the project directory, run:

```bat
docker compose config
docker compose up --build -d
docker compose ps
```

Open the Streamlit interface at http://localhost:8501.

The API is published on `127.0.0.1:8000`, so it is accessible from the host machine but not directly exposed on the host's external network interfaces. Inside Docker Compose, the Streamlit service reaches the API using the service hostname `credit-risk-api`.

View service logs:

```bat
docker compose logs -f credit-risk-api
docker compose logs -f credit-risk-app
```

Stop the services:

```bat
docker compose down
```

## AWS EC2 deployment notes

The Compose configuration can be used on an EC2 host with Docker Engine and the Docker Compose plugin installed.

1. Copy or clone the project onto the instance, including `credit1_risk_model.pkl`.
2. Confirm the Compose configuration with `docker compose config`.
3. Build and start the services with `docker compose up --build -d`.
4. Configure network access deliberately. Do not expose API port `8000` publicly; the current Compose file binds it to loopback.
5. If the Streamlit interface is to be accessed remotely, configure the required security-group rules and preferably place the application behind HTTPS and an authenticated reverse proxy.

Before using this application with real applicant data, add appropriate access controls, data-protection measures, monitoring, and model-governance procedures. Do not log sensitive applicant information unnecessarily.

## Model compatibility

The current saved model artifact was successfully loaded in the project's Python 3.12 environment with XGBoost 2.1.4 and scikit-learn 1.6.1. Loading the same artifact with XGBoost 3.4.1 failed during testing. Keep `requirements.txt` aligned with the library versions verified for the artifact, and retest the model whenever dependencies change.

## Responsible use

This project provides a model-based risk estimate for decision support. A prediction must not be treated as the sole basis for a lending decision. Any real-world use requires appropriate validation, fairness assessment, human review, privacy safeguards, and compliance with applicable laws and regulations.
