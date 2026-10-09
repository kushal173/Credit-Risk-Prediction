"""FastAPI service for the CreditLens credit-risk model."""

import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Optional

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, ConfigDict, Field


MODEL_PATH = Path(os.getenv("MODEL_PATH", str(Path(__file__).with_name("credit1_risk_model.pkl"))))

artifacts: dict | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model artifacts once when the API starts."""
    global artifacts
    if not MODEL_PATH.is_file():
        raise RuntimeError(
            f"Model artifact not found at '{MODEL_PATH}'. Place credit1_risk_model.pkl beside api.py "
            "or set MODEL_PATH to its location."
        )
    try:
        artifacts = joblib.load(MODEL_PATH)
    except Exception as exc:
        raise RuntimeError(f"Could not load model artifact '{MODEL_PATH}': {exc}") from exc
    yield


app = FastAPI(
    title="CreditLens Credit Risk API",
    description=(
        "REST API for the existing CreditLens Home Credit default-risk model. "
        "Predictions use the preprocessing artifacts and decision threshold saved with the model."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class ApplicantInput(BaseModel):
    """Applicant fields accepted by the existing Streamlit application."""

    model_config = ConfigDict(extra="forbid")

    amt_income_total: Optional[float] = Field(default=None, ge=0, description="Annual income")
    amt_credit: Optional[float] = Field(default=None, ge=0, description="Credit amount")
    amt_annuity: Optional[float] = Field(default=None, ge=0, description="Annuity amount")
    amt_goods_price: Optional[float] = Field(default=None, ge=0, description="Goods price")
    age_years: Optional[int] = Field(default=None, ge=18, le=100)
    employment_years: Optional[float] = Field(default=None, ge=0, le=60)
    cnt_children: Optional[int] = Field(default=None, ge=0, le=20)
    cnt_fam_members: Optional[int] = Field(default=None, ge=1, le=20)

    code_gender: Optional[str] = None
    name_contract_type: Optional[str] = None
    flag_own_car: Optional[str] = None
    flag_own_realty: Optional[str] = None
    name_income_type: Optional[str] = None
    name_education_type: Optional[str] = None
    name_family_status: Optional[str] = None
    name_housing_type: Optional[str] = None

    ext_source_1: Optional[float] = Field(default=None, ge=0, le=1)
    ext_source_2: Optional[float] = Field(default=None, ge=0, le=1)
    ext_source_3: Optional[float] = Field(default=None, ge=0, le=1)


def none_to_nan(value):
    return np.nan if value is None else value


def prepare_input(values: dict, model_artifacts: dict) -> pd.DataFrame:
    """Apply the same feature engineering and fitted preprocessing as the original app."""
    numerical_cols = model_artifacts["numerical_cols"]
    categorical_cols = model_artifacts["categorical_cols"]

    data = pd.DataFrame([{key: none_to_nan(value) for key, value in values.items()}])

    # Feature engineering: financial ratios and external-source summaries.
    data["credit_income_ratio"] = data["amt_credit"] / (data["amt_income_total"] + 1)
    data["annuity_income_ratio"] = data["amt_annuity"] / (data["amt_income_total"] + 1)
    data["credit_goods_ratio"] = data["amt_credit"] / (data["amt_goods_price"] + 1)
    data["income_per_person"] = data["amt_income_total"] / (data["cnt_fam_members"] + 1)
    ext_cols = ["ext_source_1", "ext_source_2", "ext_source_3"]
    data["ext_source_mean"] = data[ext_cols].mean(axis=1)
    data["ext_source_min"] = data[ext_cols].min(axis=1)

    # Match notebook property-group engineering. The current API schema does not
    # collect these component fields, so absent components become NaN and the
    # fitted numerical imputer handles the resulting combined feature.
    property_groups = model_artifacts.get("property_groups", {})
    property_component_cols = []
    for new_col, cols in property_groups.items():
        property_component_cols.extend(cols)
        for col in cols:
            if col not in data.columns:
                data[col] = np.nan
        data[f"{new_col}_combined"] = data[cols].mean(axis=1)

    # The notebook drops the original property components after creating groups.
    data.drop(columns=property_component_cols, errors="ignore", inplace=True)

    for col in categorical_cols + numerical_cols:
        if col not in data.columns:
            data[col] = np.nan

    if "employment_years" in data.columns:
        data.loc[data["employment_years"] > 100, "employment_years"] = np.nan

    data[numerical_cols] = model_artifacts["num_imputer"].transform(data[numerical_cols])
    if categorical_cols:
        data[categorical_cols] = model_artifacts["cat_imputer"].transform(data[categorical_cols])

    valid_skewed = [col for col in model_artifacts["skewed_cols"] if col in data.columns]
    if valid_skewed:
        data[valid_skewed] = data[valid_skewed].clip(
            lower=model_artifacts["lower_caps"][valid_skewed],
            upper=model_artifacts["upper_caps"][valid_skewed],
            axis=1,
        )

    if categorical_cols:
        encoded = model_artifacts["encoder"].transform(data[categorical_cols])
        if hasattr(encoded, "toarray"):
            encoded = encoded.toarray()
        encoded_df = pd.DataFrame(
            encoded,
            columns=model_artifacts["encoder"].get_feature_names_out(categorical_cols),
            index=data.index,
        )
    else:
        encoded_df = pd.DataFrame(index=data.index)

    processed = pd.concat([data.drop(columns=categorical_cols, errors="ignore"), encoded_df], axis=1)
    valid_numerical = [col for col in numerical_cols if col in processed.columns]
    if valid_numerical:
        processed[valid_numerical] = model_artifacts["scaler"].transform(processed[valid_numerical])

    return processed.reindex(columns=model_artifacts["feature_columns"], fill_value=0)


@app.get("/", tags=["Service"])
def root() -> dict:
    return {
        "service": "CreditLens Credit Risk API",
        "status": "running",
        "docs": "/docs",
        "prediction_endpoint": "POST /predict",
    }


@app.get("/health", tags=["Service"])
def health() -> dict:
    return {"status": "healthy" if artifacts is not None else "starting", "model_loaded": artifacts is not None}


@app.get("/metadata", tags=["Service"])
def metadata() -> dict:
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded yet.")
    return {
        "model_loaded": True,
        "threshold": float(artifacts["threshold"]),
        "feature_count": len(artifacts["feature_columns"]),
    }


@app.post("/predict", tags=["Prediction"])
def predict(applicant: ApplicantInput) -> dict:
    if artifacts is None:
        raise HTTPException(status_code=503, detail="Model is not loaded. Check API startup logs.")

    values = applicant.model_dump()
    supplied = sum(value is not None for value in values.values())
    if supplied == 0:
        raise HTTPException(status_code=422, detail="Provide at least one applicant field.")
    if values["amt_income_total"] is not None and values["amt_income_total"] <= 0:
        raise HTTPException(status_code=422, detail="Annual income must be greater than zero when provided.")

    try:
        processed = prepare_input(values, artifacts)
        probability = float(artifacts["model"].predict_proba(processed)[0, 1])
    except Exception as exc:
        # Keep server internals in logs while returning a clear, non-sensitive API error.
        import logging
        logging.exception("Credit risk prediction failed")
        raise HTTPException(status_code=500, detail="Prediction failed while processing the applicant data.") from exc

    threshold = float(artifacts["threshold"])
    prediction = int(probability >= threshold)
    if probability < 0.15:
        risk_tier = "Low Risk"
        recommendation = "Proceed with standard underwriting review"
    elif probability < threshold:
        risk_tier = "Medium Risk"
        recommendation = "Route for enhanced manual review"
    else:
        risk_tier = "High Risk"
        recommendation = "Escalate or decline according to lending policy"

    return {
        "default_probability": probability,
        "prediction": prediction,
        "model_outcome": "Likely default" if prediction else "Likely no default",
        "risk_tier": risk_tier,
        "recommendation": recommendation,
        "threshold": threshold,
        "supplied_fields": supplied,
        "total_fields": len(values),
        "feature_count": int(processed.shape[1]),
    }
