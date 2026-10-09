import os

import requests
import streamlit as st


st.set_page_config(
    page_title="CreditLens | Credit Risk Assessment",
    page_icon="💳",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    .stApp { background: #f5f7fb; }
    .block-container { max-width: 1280px; padding-top: 1.5rem; padding-bottom: 3rem; }
    [data-testid="stSidebar"] { background: #0f172a; }
    [data-testid="stSidebar"] * { color: #e2e8f0; }
    .hero {
        padding: 1.6rem 1.8rem; border-radius: 18px;
        background: linear-gradient(120deg, #0f172a 0%, #1e3a8a 58%, #2563eb 100%);
        color: white; box-shadow: 0 12px 30px rgba(15, 23, 42, .16); margin-bottom: 1.2rem;
    }
    .hero h1 { margin: 0; font-size: 2.15rem; letter-spacing: -.03em; }
    .hero p { margin: .45rem 0 0; color: #dbeafe; }
    .eyebrow { font-size: .78rem; font-weight: 700; letter-spacing: .13em; text-transform: uppercase; color: #93c5fd; }
    div[data-testid="stMetric"] {
        background: white; border: 1px solid #e2e8f0; border-radius: 14px;
        padding: 1rem 1.1rem; box-shadow: 0 5px 18px rgba(15, 23, 42, .05);
    }
    div[data-testid="stMetricLabel"] { color: #64748b; }
    div[data-testid="stForm"] {
        background: white; border: 1px solid #e2e8f0; border-radius: 18px;
        padding: 1.2rem 1.3rem 1.4rem; box-shadow: 0 5px 22px rgba(15, 23, 42, .05);
    }
    .section-note { color: #64748b; font-size: .9rem; margin-top: -.6rem; margin-bottom: .8rem; }
    .result-card { padding: 1rem 1.2rem; border-radius: 14px; margin-top: .8rem; font-weight: 600; }
    .low { background: #ecfdf5; color: #065f46; border: 1px solid #a7f3d0; }
    .medium { background: #fffbeb; color: #92400e; border: 1px solid #fde68a; }
    .high { background: #fef2f2; color: #991b1b; border: 1px solid #fecaca; }
    .footer { text-align: center; color: #94a3b8; font-size: .78rem; margin-top: 2rem; }
    .stButton > button, .stFormSubmitButton > button { border-radius: 10px; min-height: 3rem; font-weight: 700; }
    </style>
    """,
    unsafe_allow_html=True,
)


API_BASE_URL = os.getenv("CREDIT_RISK_API_URL", "http://127.0.0.1:8000").rstrip("/")
FALLBACK_THRESHOLD = float(os.getenv("MODEL_THRESHOLD", "0.60"))


@st.cache_data(ttl=30, show_spinner=False)
def get_api_metadata():
    response = requests.get(f"{API_BASE_URL}/metadata", timeout=3)
    response.raise_for_status()
    return response.json()


try:
    api_metadata = get_api_metadata()
    threshold = float(api_metadata["threshold"])
except (requests.RequestException, ValueError, KeyError):
    # Keep the UI available if the API is temporarily starting; predictions still require the API.
    threshold = FALLBACK_THRESHOLD


with st.sidebar:
    st.markdown("## 💳 CreditLens")
    st.caption("Decision-support dashboard")
    st.markdown("---")
    st.markdown("### How it works")
    st.markdown("1. Enter the available applicant data.\n2. Submit the assessment.\n3. Review probability and risk tier.")
    st.markdown("---")
    st.markdown("### Risk policy")
    st.markdown(f"🟢 **Low:** below 15%  \n🟠 **Medium:** 15%–{threshold:.1%}  \n🔴 **High:** {threshold:.1%} or above")
    st.markdown("---")
    st.caption("Missing fields are handled by the preprocessing pipeline. Results support—not replace—human lending decisions.")

st.markdown(
    """
    <div class="hero">
      <div class="eyebrow">ML-powered decision support</div>
      <h1>Credit Risk Assessment</h1>
      <p>Estimate an applicant's probability of default using financial, personal and external credit indicators.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

with st.form("risk_form", clear_on_submit=False):
    st.subheader("Financial profile")
    st.markdown('<div class="section-note">Enter all financial amounts in the same currency (for example, INR).</div>', unsafe_allow_html=True)
    c1, c2, c3, c4 = st.columns(4)
    income = c1.number_input("Annual income", min_value=0.0, value=None, step=10000.0, placeholder="e.g. 180000")
    credit = c2.number_input("Credit amount", min_value=0.0, value=None, step=10000.0, placeholder="e.g. 500000")
    annuity = c3.number_input("Annuity amount", min_value=0.0, value=None, step=1000.0, placeholder="e.g. 25000")
    goods_price = c4.number_input("Goods price", min_value=0.0, value=None, step=10000.0, placeholder="e.g. 450000")

    st.divider()
    st.subheader("Applicant details")
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Age", min_value=18, max_value=100, value=None, placeholder="Years")
    employment_years = c2.number_input("Employment years", min_value=0.0, max_value=60.0, value=None, placeholder="Years")
    children = c3.number_input("Number of children", min_value=0, max_value=20, value=None)
    family_members = c4.number_input("Family members", min_value=1, max_value=20, value=None)

    c1, c2, c3, c4 = st.columns(4)
    gender = c1.selectbox("Gender", [None, "F", "M"], format_func=lambda x: "Not provided" if x is None else x)
    contract_type = c2.selectbox("Contract type", [None, "Cash loans", "Revolving loans"], format_func=lambda x: "Not provided" if x is None else x)
    own_car = c3.selectbox("Owns a car", [None, "Y", "N"], format_func=lambda x: "Not provided" if x is None else x)
    own_realty = c4.selectbox("Owns property", [None, "Y", "N"], format_func=lambda x: "Not provided" if x is None else x)

    c1, c2, c3, c4 = st.columns(4)
    income_type = c1.selectbox("Income type", [None, "Working", "Commercial associate", "Pensioner", "State servant", "Businessman", "Student", "Unemployed", "Maternity leave"], format_func=lambda x: "Not provided" if x is None else x)
    education = c2.selectbox("Education", [None, "Secondary / secondary special", "Higher education", "Incomplete higher", "Lower secondary", "Academic degree"], format_func=lambda x: "Not provided" if x is None else x)
    family_status = c3.selectbox("Family status", [None, "Married", "Single / not married", "Civil marriage", "Separated", "Widow"], format_func=lambda x: "Not provided" if x is None else x)
    housing_type = c4.selectbox("Housing type", [None, "House / apartment", "Rented apartment", "With parents", "Municipal apartment", "Office apartment", "Co-op apartment"], format_func=lambda x: "Not provided" if x is None else x)

    st.divider()
    st.subheader("External credit indicators")
    st.markdown('<div class="section-note">Optional normalized scores between 0 and 1.</div>', unsafe_allow_html=True)
    c1, c2, c3 = st.columns(3)
    ext_source_1 = c1.number_input("External score 1", min_value=0.0, max_value=1.0, value=None, step=0.01, format="%.3f")
    ext_source_2 = c2.number_input("External score 2", min_value=0.0, max_value=1.0, value=None, step=0.01, format="%.3f")
    ext_source_3 = c3.number_input("External score 3", min_value=0.0, max_value=1.0, value=None, step=0.01, format="%.3f")

    submitted = st.form_submit_button("Run credit risk assessment →", type="primary", use_container_width=True)

if submitted:
    values = {
        "amt_income_total": income, "amt_credit": credit, "amt_annuity": annuity,
        "amt_goods_price": goods_price, "age_years": age,
        "employment_years": employment_years, "cnt_children": children,
        "cnt_fam_members": family_members, "code_gender": gender,
        "name_contract_type": contract_type, "flag_own_car": own_car,
        "flag_own_realty": own_realty, "name_income_type": income_type,
        "name_education_type": education, "name_family_status": family_status,
        "name_housing_type": housing_type, "ext_source_1": ext_source_1,
        "ext_source_2": ext_source_2, "ext_source_3": ext_source_3,
    }

    supplied = sum(value is not None for value in values.values())
    if supplied == 0:
        st.warning("Enter at least some applicant information before running an assessment.")
        st.stop()
    if income is not None and income <= 0:
        st.warning("Annual income must be greater than zero when provided.")
        st.stop()

    try:
        with st.spinner("Assessing applicant risk…"):
            response = requests.post(f"{API_BASE_URL}/predict", json=values, timeout=60)
            if response.status_code >= 400:
                try:
                    detail = response.json().get("detail", "The API rejected the request.")
                except ValueError:
                    detail = "The API rejected the request."
                st.error(f"Prediction API error ({response.status_code}): {detail}")
                st.stop()
            result = response.json()
    except requests.Timeout:
        st.error("The prediction API took too long to respond. Please try again.")
        st.stop()
    except requests.ConnectionError:
        st.error(
            f"Cannot connect to the prediction API at {API_BASE_URL}. "
            "Start the FastAPI service and try again."
        )
        st.stop()
    except requests.RequestException as exc:
        st.error(f"Could not complete the prediction request: {exc}")
        st.stop()

    probability = float(result["default_probability"])
    prediction = int(result["prediction"])
    risk_tier = result["risk_tier"]
    recommendation = result["recommendation"]
    risk_class = {"Low Risk": "low", "Medium Risk": "medium", "High Risk": "high"}.get(risk_tier, "medium")

    st.markdown("## Assessment result")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Default probability", f"{probability:.2%}")
    c2.metric("Risk tier", risk_tier)
    c3.metric("Model outcome", "Likely default" if prediction else "Likely no default")
    c4.metric("Data supplied", f"{result['supplied_fields']} / {result['total_fields']} fields")
    st.progress(min(max(probability, 0.0), 1.0), text=f"Estimated default risk: {probability:.2%}")
    st.markdown(f'<div class="result-card {risk_class}">{recommendation}</div>', unsafe_allow_html=True)

    with st.expander("Technical assessment details"):
        st.write(f"Decision threshold: **{threshold:.2%}**")

st.markdown('<div class="footer">CreditLens · Explainable decision support for responsible lending</div>', unsafe_allow_html=True)
