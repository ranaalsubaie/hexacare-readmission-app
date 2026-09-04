import streamlit as st
import pickle
import pandas as pd
import plotly.graph_objects as go

st.set_page_config(
    page_title="HexaCare — Readmission Risk",
    page_icon=":material/monitor_heart:",
    layout="wide",
)

# ============================================================
# Theme
# ============================================================
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.block-container { padding-top: 1.5rem; max-width: 1150px; }

/* Hero banner */
.hero {
    background: linear-gradient(120deg, #0E4C75 0%, #1C6EA4 55%, #4FA6D8 100%);
    border-radius: 18px;
    padding: 34px 40px;
    margin-bottom: 28px;
    box-shadow: 0 8px 24px rgba(28, 110, 164, 0.25);
}
.hero h1 {
    color: #FFFFFF !important;
    font-size: 2.1rem;
    font-weight: 700;
    margin-bottom: 4px;
    letter-spacing: -0.5px;
}
.hero p {
    color: #DCEEFA;
    font-size: 1.02rem;
    margin: 0;
}

/* Section cards */
div[data-testid="stVerticalBlockBorderWrapper"] {
    border-radius: 14px !important;
    border: 1px solid #DCEBF5 !important;
    box-shadow: 0 2px 10px rgba(16, 55, 92, 0.05);
    padding: 6px 4px;
}

h2, h3 { color: #10375C; font-weight: 600; letter-spacing: -0.2px; }

/* Metric cards */
[data-testid="stMetric"] {
    background: linear-gradient(180deg, #F2F9FD 0%, #E6F2FA 100%);
    border: 1px solid #CDE6F5;
    border-radius: 12px;
    padding: 16px 18px;
}
[data-testid="stMetricLabel"] { font-size: 0.82rem; color: #3E6E8E; font-weight: 500; }
[data-testid="stMetricValue"] { color: #10375C; font-weight: 700; }

/* Primary button */
button[kind="primary"] {
    background: linear-gradient(120deg, #1C6EA4, #0E4C75) !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 0.6rem 1.2rem !important;
    box-shadow: 0 4px 14px rgba(28, 110, 164, 0.35);
}
button[kind="primary"]:hover { filter: brightness(1.08); }

/* Sidebar */
[data-testid="stSidebar"] {
    background-color: #0E4C75;
}
[data-testid="stSidebar"] * { color: #EAF4FC !important; }
[data-testid="stSidebar"] hr { border-color: rgba(255,255,255,0.2) !important; }

hr { border-color: #DCEBF5 !important; }
</style>
""", unsafe_allow_html=True)

# ============================================================
# Load model
# ============================================================
@st.cache_resource
def load_model():
    with open("hexacare_readmission_pipeline.pkl", "rb") as file:
        return pickle.load(file)

try:
    bundle = load_model()
    pipeline = bundle["pipeline"]
    threshold = bundle["threshold"]

    preprocessor = pipeline.named_steps["preprocess"]
    encoder = preprocessor.named_transformers_["categorical"]

    category_options = {
        feature: list(categories)
        for feature, categories in zip(bundle["categorical_features"], encoder.categories_)
    }

except FileNotFoundError:
    st.error(
        "Model file `hexacare_readmission_pipeline.pkl` was not found in this folder. "
        "Make sure it's saved next to `app.py`, then rerun the app.",
        icon=":material/error:",
    )
    st.stop()

except (KeyError, AttributeError) as e:
    st.error(
        f"The model file loaded, but its structure doesn't match what this app expects "
        f"(missing: {e}).",
        icon=":material/error:",
    )
    st.stop()

# ============================================================
# Sidebar
# ============================================================
with st.sidebar:
    st.markdown("### :material/monitor_heart: HexaCare")
    st.caption("Diabetic Readmission Risk Platform")
    st.divider()
    with st.expander(":material/info: About this tool"):
        st.write(
            "Estimates the probability that a diabetic patient will be "
            "readmitted to hospital within 30 days of discharge, based on "
            "clinical and administrative data from their encounter."
        )

# ============================================================
# Hero header
# ============================================================
st.markdown("""
<div class="hero">
    <h1>HexaCare 30-Day Readmission Risk</h1>
    <p>Enter the patient's clinical profile below to estimate their probability of hospital readmission within 30 days.</p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# 1. Patient Information
# ============================================================
with st.container(border=True):
    st.markdown("#### :material/person: Patient Information")
    col1, col2, col3 = st.columns(3)
    with col1:
        race = st.selectbox("Race", category_options["race"])
    with col2:
        gender = st.selectbox("Gender", category_options["gender"])
    with col3:
        age = st.selectbox("Age", category_options["age"])

st.write("")

# ============================================================
# 2. Hospital Stay
# ============================================================
with st.container(border=True):
    st.markdown("#### :material/local_hospital: Hospital Stay")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        time_in_hospital = st.number_input("Time in Hospital (days)", min_value=1, value=4)
    with col2:
        num_lab_procedures = st.number_input("Lab Procedures", min_value=0, value=40)
    with col3:
        num_procedures = st.number_input("Procedures", min_value=0, value=1)
    with col4:
        num_medications = st.number_input("Medications", min_value=0, value=15)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        number_outpatient = st.number_input("Prior Outpatient Visits", min_value=0, value=0)
    with col2:
        number_emergency = st.number_input("Prior Emergency Visits", min_value=0, value=0)
    with col3:
        number_inpatient = st.number_input("Prior Inpatient Visits", min_value=0, value=0)
    with col4:
        number_diagnoses = st.number_input("Number of Diagnoses", min_value=0, value=7)

st.write("")

# ============================================================
# 3. Lab Results
# ============================================================
with st.container(border=True):
    st.markdown("#### :material/biotech: Lab Results")
    col1, col2 = st.columns(2)
    with col1:
        max_glu_serum = st.selectbox("Maximum Glucose Serum", category_options["max_glu_serum"])
    with col2:
        A1Cresult = st.selectbox("A1C Result", category_options["A1Cresult"])

st.write("")

# ============================================================
# 4. Admission Information
# ============================================================
with st.container(border=True):
    st.markdown("#### :material/emergency: Admission Information")
    col1, col2 = st.columns(2)
    with col1:
        admission_type = st.selectbox("Admission Type", category_options["admission_type"])
    with col2:
        admission_source = st.selectbox("Admission Source", category_options["admission_source"])
    discharge_disposition = st.selectbox("Discharge Disposition", category_options["discharge_disposition"])

st.write("")

# ============================================================
# 5. Diagnosis Groups
# ============================================================
with st.container(border=True):
    st.markdown("#### :material/stethoscope: Diagnosis Groups")
    col1, col2, col3 = st.columns(3)
    with col1:
        diag_1_group = st.selectbox("Primary Diagnosis", category_options["diag_1_group"])
    with col2:
        diag_2_group = st.selectbox("Secondary Diagnosis", category_options["diag_2_group"])
    with col3:
        diag_3_group = st.selectbox("Third Diagnosis", category_options["diag_3_group"])

st.write("")

# ============================================================
# 6. Diabetes Medications
# ============================================================
with st.container(border=True):
    st.markdown("#### :material/medication: Diabetes Medications")
    medication_features = [
        "metformin", "repaglinide", "nateglinide", "chlorpropamide", "glimepiride",
        "glipizide", "glyburide", "tolbutamide", "pioglitazone", "rosiglitazone",
        "acarbose", "miglitol", "tolazamide", "insulin",
        "glyburide-metformin", "glipizide-metformin",
    ]
    medication_values = {}
    cols = st.columns(4)
    for i, medication in enumerate(medication_features):
        with cols[i % 4]:
            medication_values[medication] = st.selectbox(
                medication.replace("-", " ").title(),
                category_options[medication],
                key=medication,
            )

st.write("")

# ============================================================
# 7. Medication Status
# ============================================================
with st.container(border=True):
    st.markdown("#### :material/edit_note: Medication Status")
    col1, col2 = st.columns(2)
    with col1:
        change = st.selectbox("Medication Change", category_options["change"])
    with col2:
        diabetesMed = st.selectbox("Diabetes Medication Prescribed", category_options["diabetesMed"])

st.write("")
st.write("")

_, mid, _ = st.columns([1, 2, 1])
with mid:
        predict_clicked = st.button(
        "Predict Readmission Risk",
        type="primary",
        width="stretch",
        icon=":material/monitor_heart:",
    )
# ============================================================
# Prediction
# ============================================================
if predict_clicked:

    patient = {
        "race": race, "gender": gender, "age": age,
        "time_in_hospital": time_in_hospital,
        "num_lab_procedures": num_lab_procedures,
        "num_procedures": num_procedures,
        "num_medications": num_medications,
        "number_outpatient": number_outpatient,
        "number_emergency": number_emergency,
        "number_inpatient": number_inpatient,
        "number_diagnoses": number_diagnoses,
        "max_glu_serum": max_glu_serum,
        "A1Cresult": A1Cresult,
        **medication_values,
        "change": change,
        "diabetesMed": diabetesMed,
        "admission_type": admission_type,
        "admission_source": admission_source,
        "discharge_disposition": discharge_disposition,
        "diag_1_group": diag_1_group,
        "diag_2_group": diag_2_group,
        "diag_3_group": diag_3_group,
    }

    try:
        patient_df = pd.DataFrame([patient])
        patient_df = patient_df[bundle["feature_order"]]
        probability = pipeline.predict_proba(patient_df)[0, 1]
        prediction = probability >= threshold

    except KeyError as e:
        st.error(f"A required feature is missing from the input: {e}.", icon=":material/error:")
        st.stop()
    except ValueError as e:
        st.error(f"The model couldn't process this input ({e}).", icon=":material/error:")
        st.stop()

    st.write("")
    with st.container(border=True):
        st.markdown("#### :material/monitor_heart: Prediction Result")

        gauge_col, text_col = st.columns([1.1, 1])

        with gauge_col:
            fig = go.Figure(go.Indicator(
                mode="gauge+number",
                value=probability * 100,
                number={"suffix": "%", "font": {"size": 42, "color": "#10375C"}},
                gauge={
                    "axis": {"range": [0, 100], "tickcolor": "#10375C"},
                    "bar": {"color": "#1C6EA4", "thickness": 0.3},
                    "bgcolor": "white",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 20], "color": "#DFF3E3"},
                        {"range": [20, 40], "color": "#FFF3CD"},
                        {"range": [40, 100], "color": "#FADBD8"},
                    ],
                    "threshold": {
                        "line": {"color": "#C0392B", "width": 3},
                        "thickness": 0.85,
                        "value": threshold * 100,
                    },
                },
            ))
            fig.update_layout(height=280, margin=dict(l=30, r=30, t=30, b=20))
            st.plotly_chart(fig, width="stretch")

        with text_col:
            st.metric("30-Day Readmission Risk", f"{probability:.1%}")
            st.metric("Decision Threshold", f"{threshold:.1%}")

            if prediction:
                st.error(
                    "Higher risk of 30-day readmission",
                    icon=":material/warning:",
                )
            else:
                st.success(
                    "Lower risk of 30-day readmission.",
                    icon=":material/check_circle:",
                )